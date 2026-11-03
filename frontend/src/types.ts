export type Page = "overview" | "chunking" | "embedding" | "vision";
export interface Experiment {
  id: string;
  name: string;
  description: string;
  status: "ready" | "planned";
  category: string;
  tag: string;
}
export interface Parameter {
  name: string;
  type: string;
  default: number | string;
  label: string;
  min?: number;
  max?: number;
}
export interface Strategy {
  id: string;
  name: string;
  description: string;
  rating: number;
  params: Parameter[];
}
export interface Model {
  id: string;
  name: string;
  description: string;
  dimension: number;
  provider: string;
  enabled: boolean;
}
export interface ChunkResult {
  strategy: string;
  chunks: { content: string; title?: string; type?: string }[];
  stats: {
    count: number;
    total_chars: number;
    average_chars: number;
    elapsed_ms: number;
  };
}
export interface EmbeddingResult {
  model: string;
  model_name: string;
  texts: string[];
  query: string;
  dimension: number;
  count: number;
  elapsed_ms: number;
  rankings: { index: number; text: string; score: number }[];
}
