import sys
import importlib
import warnings

def check_dependencies():
    """
    检查项目所需的所有第三方依赖是否已正确安装
    """
    required_packages = {
        # Core data science & ML
        'numpy': 'numpy',
        'sentence_transformers': 'sentence-transformers',
        'sklearn': 'scikit-learn',
        'torch': 'torch',
        'transformers': 'transformers',
        'nltk': 'nltk',
        
        # Tokenization & Metrics
        'tiktoken': 'tiktoken',
        'rouge_score': 'rouge-score',
        'bert_score': 'bert-score',
        
        # LLM Clients
        'openai': 'openai',
        'litellm': 'litellm',
        
        # Utilities
        'tqdm': 'tqdm',
        'pandas': 'pandas',
        'dotenv': 'python-dotenv',
        'tenacity': 'tenacity',
        'requests': 'requests',
        
        # GraphRAG Community
        'networkx': 'networkx',
        'leidenalg': 'leidenalg',
        'igraph': 'igraph'
    }

    missing_packages = []
    
    print("="*50)
    print("🔍 开始检查项目依赖完整性...")
    print("="*50)

    for module_name, pip_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"✅ 已安装: {pip_name} (导入名: {module_name})")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"❌ 未安装: {pip_name} (缺少模块: {module_name})")
            
    print("="*50)
    
    if missing_packages:
        print("\n⚠️ 发现缺失的依赖包！请运行以下命令安装：")
        print(f"\npip install {' '.join(missing_packages)}\n")
        print("或者直接安装 requirements.txt：")
        print("pip install -r requirements.txt\n")
        sys.exit(1)
    else:
        print("\n🎉 所有必须的依赖包均已正确安装！环境完整。")
        sys.exit(0)

if __name__ == "__main__":
    check_dependencies()
