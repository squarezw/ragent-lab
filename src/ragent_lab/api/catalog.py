"""Experiment registration. Add a route and frontend page for each ready experiment."""
EXPERIMENTS = [
    {'id': 'chunking', 'name': '文本分块', 'description': '探索文本边界，找到适合知识库的分块策略。', 'status': 'ready', 'category': 'RAG 基础', 'tag': '11 种策略'},
    {'id': 'embedding', 'name': 'Embedding 对比', 'description': '从语义相似度出发，比较不同向量模型的检索表现。', 'status': 'ready', 'category': 'RAG 基础', 'tag': '7 个模型'},
    {'id': 'vision', 'name': '图像理解', 'description': '多模型理解工程图纸并打分，再按理解 1:1 还原为 DXF。', 'status': 'ready', 'category': '多模态', 'tag': '9 个模型'},
]
