import streamlit as st
import re
from lab import strategies, text as default_text
# 新增导入
import csv
import os
from datetime import datetime
from streamlit_javascript import st_javascript

st.set_page_config(page_title="RAG 分段策略测试", layout="wide")

# 右上角浮动联系邮箱（用 email SVG 图标）
st.markdown(
    """
    <style>
    .contact-float {
        position: fixed;
        top: 18px;
        right: 32px;
        z-index: 9999;
        background: #fff;
        color: #333;
        border-radius: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        padding: 6px 18px;
        font-size: 15px;
        border: 1px solid #eee;
        opacity: 0.92;
        transition: opacity 0.2s;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .contact-float:hover { opacity: 1; }
    .contact-float svg { vertical-align: middle; margin-right: 2px; }
    </style>
    <div class="contact-float">
        <a href="mailto:square.zhao.wei@gmail.com" style="text-decoration:none;color:inherit;display:flex;align-items:center;gap:6px;">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="none" viewBox="0 0 24 24"><path fill="#333" d="M2 6.75A2.75 2.75 0 0 1 4.75 4h14.5A2.75 2.75 0 0 1 22 6.75v10.5A2.75 2.75 0 0 1 19.25 20H4.75A2.75 2.75 0 0 1 2 17.25V6.75Zm2.75-1.25a1.25 1.25 0 0 0-1.25 1.25v.637l8.25 5.5 8.25-5.5V6.75a1.25 1.25 0 0 0-1.25-1.25H4.75Zm15.25 3.263-7.6 5.067a.75.75 0 0 1-.8 0L3 8.763v8.487c0 .69.56 1.25 1.25 1.25h14.5c.69 0 1.25-.56 1.25-1.25V8.763Z"/></svg>
            square.zhao.wei@gmail.com
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# 隐藏 Streamlit header/footer/菜单栏，并减少顶部空白
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1rem !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# 更紧凑的主标题
st.markdown(
    "<h1 style='margin-top: 0.5rem; margin-bottom: 1.2rem;'>RAG 分段策略测试</h1>",
    unsafe_allow_html=True
)

# 初始化 session_state
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'result_strategy' not in st.session_state:
    st.session_state['result_strategy'] = None

# 统计访问者信息（IP 和 User-Agent）
user_agent = st_javascript("window.navigator.userAgent")
ip = st_javascript("fetch('https://api.ipify.org?format=text').then(r => r.text())")  # 去掉 await，变为非阻塞

def get_next_id(csv_path):
    if not os.path.exists(csv_path):
        return 1
    with open(csv_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) + 1

def save_stats(ip, user_agent, csv_path="stats.csv"):
    next_id = get_next_id(csv_path)
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([next_id, datetime.now().isoformat(), ip, user_agent])

# 只在首次获取到 IP 和 User-Agent 时记录
if user_agent and ip and not st.session_state.get("stats_logged"):
    save_stats(ip, user_agent)
    st.session_state["stats_logged"] = True

# 三列布局
col1, col2, col3 = st.columns([2, 1, 3])

with col1:
    st.subheader("原始文本")
    user_text = st.text_area("请输入文本", value=default_text, height=500)
    # 统计信息区域
    char_count = len(user_text)
    chinese_char_count = len(re.findall(r'[\u4e00-\u9fff]', user_text))
    st.markdown(f"**字符数：** {char_count}  |  **中文字数：** {chinese_char_count}")

with col2:
    st.subheader("分段策略")
    strategy_names = list(strategies.keys())
    selected_strategy = st.radio("分段策略", strategy_names, index=0)
    generate = st.button("生成")
    st.markdown("---")
    st.write("当前策略说明：")
    st.info(strategies[selected_strategy]["desc"])
    # 推荐指数星级展示
    rating = strategies[selected_strategy].get("rating", 0)
    stars = "★" * rating + "☆" * (5 - rating)
    st.markdown(f"推荐指数：<span style='color:gold;font-size:22px'>{stars}</span>", unsafe_allow_html=True)

    # 动态参数输入区域
    param_values = {}
    params = strategies[selected_strategy].get("params", [])
    if params:
        st.markdown("---")
        st.write("**参数设置：**")
        for param in params:
            key = f"param_{selected_strategy}_{param['name']}"
            if param["type"] == "int":
                param_values[param["name"]] = st.number_input(param["label"], value=param["default"], step=1, key=key)
            elif param["type"] == "float":
                param_values[param["name"]] = st.number_input(param["label"], value=param["default"], step=0.01, format="%.4f", key=key)
            elif param["type"] == "str":
                param_values[param["name"]] = st.text_input(param["label"], value=param["default"], key=key)
            # 可扩展更多类型

    # 只有点击生成时才更新结果
    if generate and user_text.strip():
        func = strategies[selected_strategy]["func"]
        try:
            # 只传递参数字典中有的参数
            if param_values:
                # 特殊处理keywords为list
                if "keywords" in param_values:
                    param_values["keywords"] = [k.strip() for k in param_values["keywords"].split(",") if k.strip()]
                result = func(user_text, **param_values)
            else:
                result = func(user_text)
            st.session_state['result'] = result
            st.session_state['result_strategy'] = selected_strategy
        except Exception as e:
            st.session_state['result'] = f"分段出错: {e}"
            st.session_state['result_strategy'] = selected_strategy

with col3:
    st.subheader("结果")
    # 只在有结果时显示
    if st.session_state['result'] is not None:
        result = st.session_state['result']
        # 错误信息
        if isinstance(result, str):
            st.error(result)
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            for i, chunk in enumerate(result):
                with st.container():
                    st.markdown(f"---\n**区块 {i+1}**")
                    st.json(chunk)
                    # 统计区块内容的字符数和中文字数
                    content = chunk.get('content', '')
                    char_count = len(content)
                    chinese_char_count = len(re.findall(r'[\u4e00-\u9fff]', content))
                    st.markdown(f"<span style='color:gray'>字符数：{char_count} | 中文字数：{chinese_char_count}</span>", unsafe_allow_html=True)
        else:
            for i, chunk in enumerate(result):
                with st.container():
                    st.markdown(f"---\n**区块 {i+1}**")
                    st.code(chunk, language=None)
                    char_count = len(chunk)
                    chinese_char_count = len(re.findall(r'[\u4e00-\u9fff]', chunk))
                    st.markdown(f"<span style='color:gray'>字符数：{char_count} | 中文字数：{chinese_char_count}</span>", unsafe_allow_html=True)
    else:
        st.info("请点击左侧‘生成’按钮") 