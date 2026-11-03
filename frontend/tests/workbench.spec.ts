import { expect, test } from '@playwright/test';

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

test('mobile navigation and vision placeholder have no horizontal overflow', async ({ page }) => {
  await page.setViewportSize({width:390,height:844});
  await page.goto('/');
  await expect(page.getByRole('heading', { name:'把想法，放进实验室。' })).toBeVisible();
  await page.getByRole('button', { name:'打开导航' }).click();
  await page.getByRole('link', { name:'图像理解' }).click();
  await expect(page.getByText('实验筹备中 · 暂未开放上传和推理')).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.getByRole('button', { name:'打开导航' }).click();
  await page.getByRole('link', { name:'文本分块', exact:true }).click();
  await expect(page.getByLabel('待分块文本')).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});
