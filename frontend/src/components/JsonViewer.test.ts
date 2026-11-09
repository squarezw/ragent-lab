import { isValidElement } from "react";
import { describe, expect, it } from "vitest";
import { highlightJson } from "./JsonViewer";

const classes = (nodes: ReturnType<typeof highlightJson>) =>
  nodes
    .filter(isValidElement)
    .map((n) => [(n.props as { className: string }).className, (n.props as { children: string }).children]);

describe("highlightJson", () => {
  it("colors keys, strings, numbers and literals and keeps the text intact", () => {
    const text = JSON.stringify({ label: "Ø19: 孔", value: -1.5e3, ok: true, x: null, n: [2] }, null, 2);
    const nodes = highlightJson(text);
    expect(classes(nodes)).toEqual([
      ["json-key", '"label"'], ["json-string", '"Ø19: 孔"'],
      ["json-key", '"value"'], ["json-number", "-1500"],
      ["json-key", '"ok"'], ["json-literal", "true"],
      ["json-key", '"x"'], ["json-literal", "null"],
      ["json-key", '"n"'], ["json-number", "2"],
    ]);
    const flat = nodes
      .map((n) => (isValidElement(n) ? (n.props as { children: string }).children : n))
      .join("");
    expect(flat).toBe(text);
  });

  it("does not split numbers or quotes inside strings", () => {
    const nodes = highlightJson(JSON.stringify({ a: 'say "12" now' }));
    expect(classes(nodes)).toEqual([["json-key", '"a"'], ["json-string", '"say \\"12\\" now"']]);
  });
});
