"""
Application settings and configuration.
"""

# Default text for testing chunking strategies
DEFAULT_TEXT = """关于完善中国特色现代企业制度的意见
三、完善公司治理结构
（三）健全企业产权结构。尊重企业独立法人财产权，形成归属清晰、结构合理、流转顺畅的企业产权制度。国有企业要根据功能定位逐步调整优化股权结构，形成股权结构多元、股东行为规范、内部约束有效、运行高效灵活的经营机制。鼓励民营企业构建简明、清晰、可穿透的股权结构。
（四）完善国有企业公司治理。加快建立健全权责法定、权责透明、协调运转、有效制衡的公司治理机制，强化章程在公司治理中的基础作用。党委（党组）发挥把方向、管大局、保落实的领导作用。股东会是公司的权力机构，股东按照出资比例和章程行使表决权，不得超出章程规定干涉企业日常经营。董事会发挥定战略、作决策、防风险的作用，推动集团总部授权放权与分批分类落实子企业董事会职权有机衔接，规范落实董事会向经理层授权制度。完善外部董事评价和激励约束机制，落实外部董事知情权、表决权、监督权、建议权。经理层发挥谋经营、抓落实、强管理的作用，全面推进任期制和契约化管理。鼓励国有企业参照经理层成员任期制和契约化管理方式，更大范围、分层分类落实管理人员经营管理责任。
（五）支持民营企业优化法人治理结构。鼓励民营企业根据实际情况采取合伙制、公司制等多种组织形式，完善内部治理规则，制定规范的章程，保持章程与出资协议的一致性，规范控股股东、实际控制人行为。支持引导民营企业完善治理结构和管理制度，鼓励有条件的民营企业规范组建股东会、董事会、经理层。鼓励家族企业创新管理模式、组织结构、企业文化，逐步建立现代企业制度。
（六）发挥资本市场对完善公司治理的推动作用。强化控股股东对公司的诚信义务，支持上市公司引入持股比例5%以上的机构投资者作为积极股东。严格落实上市公司独立董事制度，设置独立董事占多数的审计委员会和独立董事专门会议机制。完善上市公司治理领域信息披露制度，促进提升决策管理的科学性。
```
import numpy as np
```
八、保障措施
各地区各部门要结合实际抓好本意见贯彻落实。企业要深刻认识完善中国特色现代企业制度的重要意义，落实主体责任，以企业制度创新推动高质量发展。完善相关法律法规，推动修订企业国有资产法等，推动企业依法经营、依法治企。规范会计、咨询、法律、信用评级等专业机构执业行为，加强对专业机构的从业监管，发挥其执业监督和专业服务作用，维护公平竞争、诚信规范的良好市场环境。加强对现代企业制度实践探索和成功经验的宣传，总结一批企业党建典型经验，推广一批公司治理典型实践案例。
"""


class AppConfig:
    """Application configuration settings."""
    
    # Page configuration
    PAGE_TITLE = "RAG 分段策略测试"
    PAGE_LAYOUT = "wide"
    
    # Statistics
    STATS_FILE = "stats.csv"
    
    # UI Settings
    DEFAULT_TEXT_AREA_HEIGHT = 500
    DEFAULT_COLUMN_RATIOS = [2, 1, 3]  # col1, col2, col3
    
    # Chunking defaults
    DEFAULT_CHUNK_SIZE = 100
    DEFAULT_WINDOW_SIZE = 3
    DEFAULT_STRIDE = 2
    DEFAULT_SIMILARITY_THRESHOLD = 0.85
    DEFAULT_RECURSIVE_CHUNK_SIZE = 150
    DEFAULT_CHUNK_OVERLAP = 20
    DEFAULT_TEXT_LENGTH_THRESHOLD = 1000
    DEFAULT_KEYWORDS = "完善,电话,邮箱,###"
    DEFAULT_CHUNK_LEN_THRESHOLD = 500
    
    # GitHub repository
    GITHUB_URL = "https://github.com/squarezw/ragent-lab"
    
    @classmethod
    def get_column_ratios(cls) -> list:
        """Get column ratios for Streamlit layout."""
        return cls.DEFAULT_COLUMN_RATIOS
