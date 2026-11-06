import { expect, test, type Page } from '@playwright/test';

test('real API chunking, changing input invalidates the result', async ({ page }) => {
  await page.goto('/#chunking');
  await page.getByLabel('分块策略').selectOption('固定长度分块');
  await page.getByLabel('待分块文本').fill('abcdefghij');
  await page.getByLabel('分块长度（字符数）').fill('4');
  await page.getByRole('button', { name: '运行实验' }).click();
  await expect(page.locator('.chunk-card p')).toHaveText(['abcd','efgh','ij']);
  await page.getByLabel('待分块文本').fill('new input');
  await expect(page.locator('.chunk-card')).toHaveCount(0);
});

test('embedding runs once, comparison requires identical datasets and query', async ({ page }) => {
  let requests = 0;
  await page.route('**/api/embedding/run', async route => {
    requests++;
    const body = route.request().postDataJSON();
    await route.fulfill({ json: { model:body.model, model_name:body.model, texts:body.texts, query:body.query, dimension:1536, count:body.texts.length, elapsed_ms:123, rankings:body.texts.map((text:string,index:number)=>({text,index,score:1-index*.1})) } });
  });
  await page.goto('/#embedding');
  await page.getByLabel('API Key', { exact:true }).fill('fixture-secret');
  await page.getByRole('button', { name:'生成向量并分析' }).click();
  await expect(page.locator('.ranking-item')).toHaveCount(5);
  await page.getByRole('button', { name:'加入模型对比' }).click();
  await expect(page.locator('.comparison-card')).toHaveCount(1);
  expect(requests).toBe(1);
  await page.getByLabel('查询文本').fill('新的查询');
  await expect(page.locator('.ranking-item')).toHaveCount(0);
  await page.getByRole('button', { name:'生成向量并分析' }).click();
  await page.getByRole('button', { name:'加入模型对比' }).click();
  await expect(page.getByRole('alert')).toContainText('相同的候选文本和查询');
  expect(requests).toBe(2);
  await expect(page.locator('.comparison-card')).toHaveCount(1);
  expect(await page.evaluate(()=>JSON.stringify(localStorage))).not.toContain('fixture-secret');
});

test("mobile navigation and vision workbench have no horizontal overflow", async ({
  page,
}) => {
  await mockVision(page, { samples: [SAMPLE] });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "把想法，放进实验室。" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "打开导航" }).click();
  const visionLink = page.getByRole("link", { name: "图像理解" });
  await expect(visionLink).not.toContainText("SOON");
  await visionLink.click();
  await expect(
    page.getByRole("navigation", { name: "实验步骤" }),
  ).toBeVisible();
  await expect(page.getByRole("button", { name: "② 理解对比" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "③ 还原" })).toBeDisabled();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "打开导航" }).click();
  await page.getByRole("link", { name: "文本分块", exact: true }).click();
  await expect(page.getByLabel("待分块文本")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});

// ---- vision fixtures: every /api/vision/** call is mocked, never the real OpenRouter ----

const PNG_1x1 = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);
const SAMPLE = { id: "flange", name: "法兰盘", scorable: true };
const MODELS = [
  {
    id: "vendor/model-a",
    name: "Model A",
    family: "a",
    notes: "",
    params: { max_tokens: 8000, temperature: 0, reasoning: null },
    price: { prompt: "0.000003", completion: "0.000015" },
    gateways: ["openrouter", "kie"],
  },
  {
    id: "vendor/model-b",
    name: "Model B",
    family: "b",
    notes: "",
    params: { max_tokens: 8000, temperature: 0, reasoning: null },
    price: null,
    gateways: ["openrouter"],
  },
];
const META = {
  model: "x",
  provider: null,
  finish_reason: "stop",
  usage: {},
  cost: 0.0123,
  elapsed_ms: 4200,
  params: { max_tokens: 8000, temperature: 0, reasoning: null },
};
const UNDERSTANDING = {
  part: {
    name: "法兰盘",
    drawing_number: "D-1",
    material: "Q235",
    scale: "1:1",
    notes: [],
  },
  views: [],
  features: [],
  dimensions: [],
  relations: [],
  derived: [],
  unresolved: [],
};
const GEOMETRY = {
  units: "mm",
  sheet: { width: 100, height: 80 },
  entities: [
    {
      type: "line",
      layer: "OUTLINE",
      dimension_ref: null,
      start: [0, 0],
      end: [50, 0],
    },
  ],
};
const SVG =
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 80"><rect x="10" y="10" width="80" height="60" fill="none" stroke="black"/></svg>';

type Job = {
  kind: string;
  model?: string;
  gateway?: string;
  polls: number;
  fail?: string;
};

async function mockVision(
  page: Page,
  opts: { samples: unknown[]; failOnce?: string; uploadGoneOnce?: boolean },
) {
  const jobs = new Map<string, Job>();
  let uploads = 0;
  let goneServed = false;
  const failed = new Set<string>();
  let seq = 0;
  const addJob = (job: Omit<Job, "polls">) => {
    const id = `job-${++seq}`;
    jobs.set(id, { ...job, polls: 0 });
    return { json: { job_id: id } };
  };
  const result = (job: Job) => {
    const name = MODELS.find((m) => m.id === job.model)?.name ?? job.model;
    switch (job.kind) {
      case "understand":
        return {
          model: job.model,
          model_name: name,
          understanding: UNDERSTANDING,
          meta:
            job.gateway === "kie"
              ? { ...META, gateway: "kie", cost: null, credits: 0.28 }
              : META,
        };
      case "score":
        return {
          score: 75,
          counts: { correct: 3, uncertain: 0, wrong: 1 },
          meta: META,
          judgements: [
            {
              item_id: "D1",
              verdict: "correct",
              evidence: "外径 Ø120",
              reason: "一致",
            },
          ],
        };
      case "consensus":
        return {
          meta: META,
          groups: [
            {
              label: "外径",
              view: "主视图",
              interpretations: [
                { summary: "Ø120", models: ["Model A", "Model B"] },
              ],
            },
          ],
          singletons: [
            { label: "倒角 C1", models: ["Model B"], summary: "只有一处倒角" },
          ],
        };
      default:
        return {
          geometry: GEOMETRY,
          svg: SVG,
          skipped: [],
          meta: META,
          warnings: [
            {
              entity_index: 0,
              dimension_id: "D1",
              expected: 120,
              actual: 118,
              message: "尺寸 D1 与图形不符",
            },
            {
              entity_index: 0,
              dimension_id: "D9",
              expected: 0,
              actual: 0,
              message: "引用了不存在的尺寸 D9",
            },
          ],
        };
    }
  };
  await page.route("**/api/vision/**", async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname.replace("/api/vision", "");
    const body =
      req.method() === "POST" && path !== "/images" ? req.postDataJSON() : null;
    if (path === "/models") return route.fulfill({ json: MODELS });
    if (path === "/samples") return route.fulfill({ json: opts.samples });
    if (/^\/(samples\/[^/]+\/image|images\/[^/]+)$/.test(path))
      return route.fulfill({ body: PNG_1x1, contentType: "image/png" });
    if (path === "/images")
      return route.fulfill({
        json: {
          image_id: `up-${++uploads}`,
          width: 1,
          height: 1,
          sha256: "x",
          name: "mine.png",
        },
      });
    if (path === "/understand") {
      if (opts.uploadGoneOnce && body.image.id === "up-1") {
        goneServed = true;
        return route.fulfill({
          status: 404,
          json: { detail: "图片已过期，请重新上传。" },
        });
      }
      const fail = opts.failOnce === body.model && !failed.has(body.model);
      if (fail) failed.add(body.model);
      return route.fulfill(
        addJob({
          kind: "understand",
          model: body.model,
          gateway: body.gateway,
          fail: fail ? "OpenRouter Key 无效或已撤销。" : undefined,
        }),
      );
    }
    if (path === "/score" || path === "/consensus" || path === "/generate")
      return route.fulfill(addJob({ kind: path.slice(1), model: body.model }));
    if (path === "/export")
      return route.fulfill({
        body: "0\nEOF\n",
        contentType: "application/dxf",
      });
    const m = path.match(/^\/jobs\/(.+)$/);
    if (m) {
      const job = jobs.get(m[1]);
      if (!job) return route.fulfill({ status: 404, json: { detail: "gone" } });
      job.polls += 1;
      const base = {
        id: m[1],
        kind: job.kind,
        label: "",
        created: 0,
        started: 0,
        finished: null,
        error_kind: null,
      };
      if (job.polls === 1)
        return route.fulfill({
          json: { ...base, status: "running", result: null, error: null },
        });
      if (job.fail)
        return route.fulfill({
          json: { ...base, status: "failed", result: null, error: job.fail },
        });
      return route.fulfill({
        json: { ...base, status: "done", result: result(job), error: null },
      });
    }
    return route.fulfill({ status: 404, json: { detail: `unmocked ${path}` } });
  });
}

async function startUnderstanding(page: Page) {
  await page.getByLabel("OpenRouter Key").fill("fixture-secret");
  await page.getByRole("button", { name: "下一步：理解对比" }).click();
  await page.getByRole("button", { name: "开始理解" }).click();
}

test("vision: understand two models, score, redraw and download", async ({
  page,
}) => {
  await mockVision(page, { samples: [SAMPLE] });
  await page.goto("/#vision");
  await page.getByRole("button", { name: /法兰盘/ }).click();
  await expect(page.getByRole("img", { name: /图纸预览/ })).toBeVisible();
  await expect(page.getByLabel("在本浏览器记住")).not.toBeChecked();
  const scoreReq = page.waitForRequest(
    (r) => r.url().endsWith("/api/vision/score") && r.method() === "POST",
  );
  await startUnderstanding(page);
  expect((await scoreReq).postDataJSON().judge_model).toBe("vendor/model-a");
  const cards = page.locator(".vision-model-card");
  await expect(cards).toHaveCount(2);
  await expect(cards.filter({ hasText: "75.0" })).toHaveCount(2, {
    timeout: 20000,
  });
  await expect(cards.first()).toContainText("费用");
  await expect(page.getByText("价格未知")).toBeVisible();
  await cards.first().getByText("逐项评分").click();
  await expect(cards.first().locator("table")).toContainText("外径 Ø120");
  await page.getByRole("button", { name: "③ 还原" }).click();
  await expect(page.getByLabel("理解来源")).toHaveValue("vendor/model-a");
  await expect(page.getByLabel("生成模型")).toHaveValue("vendor/model-a");
  await page.getByRole("button", { name: "生成还原图" }).click();
  const gen = page.locator(".vision-generation");
  await expect(gen).toHaveCount(1);
  await expect(gen.getByLabel("还原预览")).toBeVisible({ timeout: 20000 });
  await expect(gen.getByLabel("还原预览").locator("img")).toHaveAttribute(
    "src",
    /^data:image\/svg\+xml/,
  );
  await expect(gen).toContainText("引用了不存在的尺寸 D9");
  await expect(gen).not.toContainText("预期 0 mm");
  const download = page.waitForEvent("download");
  await gen.getByRole("button", { name: "下载 DXF" }).click();
  expect((await download).suggestedFilename()).toMatch(/\.dxf$/);
  expect(await page.evaluate(() => JSON.stringify(localStorage))).not.toContain(
    "fixture-secret",
  );
});

test("vision: one model fails without affecting others and can retry", async ({
  page,
}) => {
  await mockVision(page, { samples: [SAMPLE], failOnce: "vendor/model-b" });
  await page.goto("/#vision");
  await page.getByRole("button", { name: /法兰盘/ }).click();
  await startUnderstanding(page);
  const bad = page.locator(".vision-model-card", { hasText: "Model B" });
  const good = page.locator(".vision-model-card", { hasText: "Model A" });
  await expect(bad).toContainText("OpenRouter Key 无效或已撤销。", {
    timeout: 20000,
  });
  await expect(good).toContainText("已完成");
  await expect(good).toContainText("75.0", { timeout: 20000 });
  await bad.getByRole("button", { name: "重试", exact: true }).click();
  await expect(bad).toContainText("已完成", { timeout: 20000 });
  await expect(bad).toContainText("75.0", { timeout: 20000 });
});

test("vision: uploaded drawing shows consensus instead of score", async ({
  page,
}) => {
  await mockVision(page, { samples: [] });
  await page.goto("/#vision");
  await expect(page.getByText("暂无内置测试图，请上传你的图纸")).toBeVisible();
  await page
    .getByLabel("上传图纸")
    .setInputFiles({
      name: "mine.png",
      mimeType: "image/png",
      buffer: PNG_1x1,
    });
  await expect(page.getByRole("img", { name: /图纸预览/ })).toBeVisible();
  await startUnderstanding(page);
  await expect(
    page.locator(".vision-model-card", { hasText: "已完成" }),
  ).toHaveCount(2, { timeout: 20000 });
  await expect(page.getByText("未评分")).toHaveCount(0);
  const judge = page.getByLabel("评判模型");
  await expect(judge).toHaveValue("vendor/model-a");
  await judge.selectOption("vendor/model-b");
  const consensusReq = page.waitForRequest(
    (r) => r.url().endsWith("/api/vision/consensus") && r.method() === "POST",
  );
  await page.getByRole("button", { name: "生成一致性对照" }).click();
  expect((await consensusReq).postDataJSON().judge_model).toBe("vendor/model-b");
  expect(
    await page.evaluate(() => localStorage.getItem("vision-judge-model")),
  ).toBe("vendor/model-b");
  await expect(page.locator(".vision-consensus .minority").first()).toBeVisible(
    { timeout: 20000 },
  );
  expect(
    await page.locator(".vision-consensus .minority").count(),
  ).toBeGreaterThanOrEqual(1);
});

test("vision: Kie.ai gateway runs only the models it serves", async ({
  page,
}) => {
  await mockVision(page, { samples: [SAMPLE] });
  const bodies: { model: string; gateway: string; api_key: string }[] = [];
  page.on("request", (r) => {
    if (r.url().endsWith("/api/vision/understand") && r.method() === "POST")
      bodies.push(r.postDataJSON());
  });
  await page.goto("/#vision");
  await page.getByRole("button", { name: /法兰盘/ }).click();
  await expect(page.getByLabel("OpenRouter Key")).toBeVisible();
  await page.getByRole("radio", { name: /Kie\.ai/ }).click();
  await expect(page.getByRole("radio", { name: /Kie\.ai/ })).toHaveAttribute(
    "aria-checked",
    "true",
  );
  await page.getByLabel("Kie.ai Key").fill("kie-fixture-secret");
  await expect(page.getByText("Kie.ai · 1 个模型")).toBeVisible();
  await page.getByRole("button", { name: "下一步：理解对比" }).click();
  await expect(page.getByText("按 Kie.ai 积分计费")).toBeVisible();
  await expect(page.getByText("Model B")).toHaveCount(0);
  await page.getByRole("button", { name: "开始理解" }).click();
  const card = page.locator(".vision-model-card", { hasText: "Model A" });
  await expect(card).toContainText("0.28 积分", { timeout: 20000 });
  expect(bodies).toHaveLength(1);
  expect(bodies[0]).toMatchObject({
    model: "vendor/model-a",
    gateway: "kie",
    api_key: "kie-fixture-secret",
  });
  const stored = await page.evaluate(() => JSON.stringify(localStorage));
  expect(stored).toContain("vision-gateway");
  expect(stored).not.toContain("kie-fixture-secret");
});

test("vision: an upload the server lost is re-sent automatically", async ({
  page,
}) => {
  await mockVision(page, { samples: [], uploadGoneOnce: true });
  const images: string[] = [];
  page.on("request", (r) => {
    if (r.url().endsWith("/api/vision/understand") && r.method() === "POST")
      images.push(r.postDataJSON().image.id);
  });
  await page.goto("/#vision");
  await page.getByLabel("上传图纸").setInputFiles({
    name: "mine.png",
    mimeType: "image/png",
    buffer: PNG_1x1,
  });
  await expect(page.getByRole("img", { name: /图纸预览/ })).toBeVisible();
  await startUnderstanding(page);
  await expect(
    page.locator(".vision-model-card", { hasText: "已完成" }),
  ).toHaveCount(2, { timeout: 20000 });
  await expect(page.getByText("图片已过期")).toHaveCount(0);
  // Both models first tried the lost id; the file was uploaded once more and reused.
  expect(images.filter((id) => id === "up-2")).toHaveLength(2);
  expect(images.filter((id) => id === "up-3")).toHaveLength(0);
});
