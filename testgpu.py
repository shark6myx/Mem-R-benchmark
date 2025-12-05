import nltk
from nltk.data import find

try:
    # 查找 punkt_tab 资源的路径（NLTK 内部加载逻辑）
    punkt_tab_path = find("tokenizers/punkt_tab/english/")
    print(f"✅ 找到 punkt_tab 资源，路径：{punkt_tab_path}")

    # 再测试分词，确认使用该资源
    tokens = nltk.word_tokenize("LGBTQ support group was powerful")
    print(f"分词验证成功：{tokens}")
except LookupError as e:
    print(f"❌ 未找到 punkt_tab 资源：{e}")
except Exception as e:
    print(f"❌ 验证失败：{e}")