import streamlit as st
import re
from lab import strategies, text as default_text

st.set_page_config(page_title="分段策略实验室", layout="wide")

# 初始化 session_state
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'result_strategy' not in st.session_state:
    st.session_state['result_strategy'] = None

# 三列布局
col1, col2, col3 = st.columns([2, 1, 3])

with col1:
    st.header("输入文本")
    user_text = st.text_area("请输入文本", value=default_text, height=400)
    # 统计信息区域
    char_count = len(user_text)
    chinese_char_count = len(re.findall(r'[\u4e00-\u9fff]', user_text))
    st.markdown(f"**字符数：** {char_count}  |  **中文字数：** {chinese_char_count}")

with col2:
    st.header("选择分段策略")
    strategy_names = list(strategies.keys())
    selected_strategy = st.selectbox("分段策略", strategy_names)
    st.markdown("---")
    st.write("当前策略说明：")
    st.write(selected_strategy)
    generate = st.button("生成")

    # 只有点击生成时才更新结果
    if generate and user_text.strip():
        func = strategies[selected_strategy]
        try:
            result = func(user_text)
            st.session_state['result'] = result
            st.session_state['result_strategy'] = selected_strategy
        except Exception as e:
            st.session_state['result'] = f"分段出错: {e}"
            st.session_state['result_strategy'] = selected_strategy

with col3:
    st.header("分段结果")
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