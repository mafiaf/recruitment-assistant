import sys
import types

# stub heavy external modules so main can be imported without full dependencies
mods = {
    'mammoth': {},
    'numpy': {'array': lambda x: x},
    'openai': {},
    'pdfplumber': {},
    'PyPDF2': {},
    'docx': {'Document': lambda *a, **k: None},
    'httpx': {},
    'dotenv': {'load_dotenv': lambda: None},
    'certifi': {'where': lambda: ''},
}
for name, attrs in mods.items():
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

# stub passlib with minimal CryptContext
passlib_context = types.ModuleType('passlib.context')
passlib_context.CryptContext = lambda **kw: types.SimpleNamespace(hash=lambda p: p, verify=lambda p, h: p == h)
sys.modules['passlib.context'] = passlib_context
passlib_hash = types.ModuleType('passlib.hash')
passlib_hash.bcrypt = types.SimpleNamespace(hash=lambda p: p, verify=lambda p, h: p == h)
sys.modules['passlib.hash'] = passlib_hash

# stub Mongo and related modules used by db/mongo_utils
pymongo = types.ModuleType('pymongo')
pymongo.MongoClient = lambda *a, **k: None
pymongo.errors = types.SimpleNamespace(PyMongoError=Exception)
sys.modules['pymongo'] = pymongo
bson = types.ModuleType('bson')
bson.ObjectId = str
sys.modules['bson'] = bson

# stub custom helper modules used by main
stub_db = types.ModuleType('db')
stub_db.chat_find_one = lambda *a, **kw: None
stub_db.chat_upsert = lambda *a, **kw: None
stub_db.resumes_all = lambda: []
stub_db.resumes_by_ids = lambda ids: []
stub_db.resumes_collection = None
sys.modules['db'] = stub_db

stub_mongo_utils = types.ModuleType('mongo_utils')
stub_mongo_utils.update_resume = lambda *a, **kw: None
stub_mongo_utils.delete_resume_by_id = lambda *a, **kw: None
stub_mongo_utils.db = {}
sys.modules['mongo_utils'] = stub_mongo_utils

stub_pinecone_utils = types.ModuleType('pinecone_utils')
stub_pinecone_utils.add_resume_to_pinecone = lambda *a, **kw: None
stub_pinecone_utils.embed_text = lambda *a, **kw: []
stub_pinecone_utils.index = None
stub_pinecone_utils.search_best_resumes = lambda *a, **kw: []
sys.modules['pinecone_utils'] = stub_pinecone_utils
