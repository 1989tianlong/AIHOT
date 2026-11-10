// kimi-k2.7-code rejects temperature other than 1. chatJson omits that field on the wire, including
// when the caller or extra JSON sets one. Every other model still sends its temperature.
import { pointModels, stub, tag } from "./setup.ts";
import assert from "node:assert/strict";
import { after, test } from "node:test";
import { z } from "zod";
import { closeDb } from "@aihot/backend/db";
import { chatJson, type ChatJsonOptions } from "@aihot/backend/providers/llm";
import { stopBoss } from "@aihot/backend/jobs/queue";

const schema = z.object({ ok: z.boolean() });
const SYSTEM = "Reply with JSON.";
const hits: Array<{ url: string; body: Record<string, unknown> }> = [];
const provider = await stub((_hit, req) => {
  hits.push({ url: req.url, body: JSON.parse(req.body) as Record<string, unknown> });
  return { id: `stub-${hits.length}`, choices: [{ message: { content: JSON.stringify({ ok: true }) } }], usage: { prompt_tokens: 1, completion_tokens: 1 } };
});
Object.assign(process.env, { LLM_BASE_URL: `${provider.url}/v1`, LLM_API_KEY: "test-key", MODEL_CALLS_ENABLED: "true" });
delete process.env.LLM_EXTRA_JSON;
delete process.env.LLM_JSON_MODE;
delete process.env.LLM_REASONING_TOKENS;
pointModels(provider.url, ["deepseek-flash"]);

const T = tag();
let seq = 0;

function messages(user: string) {
  return [{ role: "system", content: SYSTEM }, { role: "user", content: user }];
}

async function send(llmModel: string, opts: Partial<ChatJsonOptions<typeof schema>> = {}) {
  seq += 1;
  const user = `payload ${seq}`;
  process.env.LLM_MODEL = llmModel;
  const before = hits.length;
  const result = await chatJson({
    model: "default", purpose: "invariant_test", subject: `llm-temp:${T}:${seq}`, promptVersion: "llm-temp-1",
    system: SYSTEM, user, schema, ...opts,
  });
  assert.equal(result.data.ok, true);
  assert.equal(hits.length, before + 1, "the call is sent once");
  const hit = hits.at(-1)!;
  assert.equal(hit.url, "/v1/chat/completions");
  return { hit: hit.body, user, again: () => chatJson({
    model: "default", purpose: "invariant_test", subject: `llm-temp:${T}:${seq}`, promptVersion: "llm-temp-1",
    system: SYSTEM, user, schema, ...opts,
  }) };
}

after(async () => {
  await provider.close();
  await stopBoss();
  await closeDb();
});

test("kimi-k2.7-code omits temperature for the default, an explicit value, and extra JSON", async () => {
  const plain = await send("kimi-k2.7-code");
  assert.deepEqual(plain.hit, {
    model: "kimi-k2.7-code", messages: messages(plain.user), max_tokens: 1500, response_format: { type: "json_object" },
  });
  assert.equal(Object.hasOwn(plain.hit, "temperature"), false);

  for (const temperature of [0, 0.2, 1]) {
    const explicit = await send("kimi-k2.7-code", { temperature, maxTokens: 2048 });
    assert.deepEqual(explicit.hit, {
      model: "kimi-k2.7-code", messages: messages(explicit.user), max_tokens: 2048, response_format: { type: "json_object" },
    });
    assert.equal(Object.hasOwn(explicit.hit, "temperature"), false, `explicit temperature ${temperature} is not sent`);
  }

  process.env.LLM_EXTRA_JSON = JSON.stringify({ temperature: 0.2, top_p: 0.9, enable_thinking: false });
  try {
    const extra = await send("kimi-k2.7-code", { temperature: 0, maxTokens: 1800 });
    assert.deepEqual(extra.hit, {
      model: "kimi-k2.7-code", messages: messages(extra.user), max_tokens: 1800, response_format: { type: "json_object" },
      top_p: 0.9, enable_thinking: false,
    });
    assert.equal(Object.hasOwn(extra.hit, "temperature"), false, "extra temperature cannot override the omission");
    const calls = hits.length;
    const reused = await extra.again();
    assert.equal(reused.reused, true);
    assert.equal(hits.length, calls, "a repeat of the same request is not sent again");
  } finally {
    delete process.env.LLM_EXTRA_JSON;
  }

  process.env.LLM_EXTRA_JSON = JSON.stringify({ temperature: 1 });
  try {
    const allowed = await send("kimi-k2.7-code", { json: false, maxTokens: 512 });
    assert.deepEqual(allowed.hit, {
      model: "kimi-k2.7-code", messages: messages(allowed.user), max_tokens: 512,
    });
    assert.equal(Object.hasOwn(allowed.hit, "temperature"), false, "extra temperature 1 is omitted with the field");
    assert.equal(Object.hasOwn(allowed.hit, "response_format"), false);
  } finally {
    delete process.env.LLM_EXTRA_JSON;
  }
});

test("other models still send temperature, including explicit 0 and an extra override", async () => {
  const plain = await send("other-model");
  assert.deepEqual(plain.hit, {
    model: "other-model", messages: messages(plain.user), temperature: 0.2, max_tokens: 1500, response_format: { type: "json_object" },
  });

  const zero = await send("kimi-k2.7-code-preview", { temperature: 0, maxTokens: 2048 });
  assert.deepEqual(zero.hit, {
    model: "kimi-k2.7-code-preview", messages: messages(zero.user), temperature: 0, max_tokens: 2048, response_format: { type: "json_object" },
  });

  process.env.LLM_EXTRA_JSON = JSON.stringify({ temperature: 0.4, top_p: 0.9 });
  try {
    const extra = await send("deepseek-chat", { temperature: 0.2 });
    assert.deepEqual(extra.hit, {
      model: "deepseek-chat", messages: messages(extra.user), temperature: 0.4, max_tokens: 1500,
      response_format: { type: "json_object" }, top_p: 0.9,
    });
  } finally {
    delete process.env.LLM_EXTRA_JSON;
  }

  seq += 1;
  const user = `payload ${seq}`;
  const before = hits.length;
  const preset = await chatJson({
    model: "deepseek-flash", purpose: "invariant_test", subject: `llm-temp:${T}:${seq}`, promptVersion: "llm-temp-1",
    system: SYSTEM, user, schema, temperature: 0.3, maxTokens: 1500,
  });
  assert.equal(preset.data.ok, true);
  assert.equal(hits.length, before + 1);
  assert.deepEqual(hits.at(-1)!.body, {
    model: "deepseek-flash", messages: messages(user), temperature: 0.3, max_tokens: 1500,
    response_format: { type: "json_object" }, thinking: { type: "disabled" },
  });
});
