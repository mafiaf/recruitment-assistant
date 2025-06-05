import sys
import types

# stub heavy external modules so main can be imported without full dependencies
mods = {
    'mammoth': {},
    'numpy': {'array': lambda x: x},
    'openai': {},
    'pdfplumber': {},
    'PyPDF2': {'PdfReader': lambda *a, **k: types.SimpleNamespace(pages=[])},
    'docx': {'Document': lambda *a, **k: None},
    'dotenv': {'load_dotenv': lambda: None},
    'certifi': {'where': lambda: ''},
    'itsdangerous': {'URLSafeTimedSerializer': lambda *a, **k: None, 'BadSignature': Exception},
}
for name, attrs in mods.items():
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

# minimal httpx stub for Starlette's TestClient
if 'httpx' not in sys.modules:
    httpx = types.ModuleType('httpx')

    class Request:
        def __init__(self, method, url):
            from urllib.parse import urlsplit
            parts = urlsplit(url)
            self.method = method
            self.url = types.SimpleNamespace(
                scheme=parts.scheme or 'http',
                netloc=parts.netloc.encode(),
                path=parts.path,
                raw_path=parts.path.encode(),
                query=parts.query.encode(),
            )
            self.headers = {}
            self._content = b''

        def read(self):
            return self._content

    class ByteStream:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

    class Response:
        def __init__(self, status_code=200, headers=None, stream=None, request=None):
            self.status_code = status_code
            self.headers = headers or []
            self.request = request
            data = stream.read() if hasattr(stream, 'read') else b''
            self.text = data.decode()
            self.content = data

    class BaseTransport:
        pass

    class Client:
        def __init__(self, *args, **kwargs):
            self._transport = kwargs.get('transport')
            self.base_url = kwargs.get('base_url', '')

        def request(self, method, url, **kwargs):
            req = Request(method, self.base_url + url)
            return self._transport.handle_request(req)

        def get(self, url, **kwargs):
            return self.request('GET', url, **kwargs)

        def post(self, url, **kwargs):
            return self.request('POST', url, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    httpx.Request = Request
    httpx.Response = Response
    httpx.ByteStream = ByteStream
    httpx.Client = Client
    httpx.BaseTransport = BaseTransport
    httpx._client = types.SimpleNamespace(
        USE_CLIENT_DEFAULT=object(),
        CookieTypes=object,
        UseClientDefault=object,
        TimeoutTypes=object,
    )
    httpx._types = types.SimpleNamespace(
        URLTypes=object,
        HeaderTypes=object,
        QueryParamTypes=object,
        CookieTypes=object,
        RequestContent=object,
        RequestFiles=object,
        AuthTypes=object,
    )

    sys.modules['httpx'] = httpx

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

# stub multipart to satisfy FastAPI's form handling
multipart = types.ModuleType('multipart')
multipart.__version__ = '0'
sub = types.ModuleType('multipart.multipart')
sub.parse_options_header = lambda *a, **kw: None
multipart.multipart = sub
sys.modules['multipart'] = multipart
sys.modules['multipart.multipart'] = sub

# minimal jinja2 stub for starlette.templating
jinja2 = types.ModuleType('jinja2')
class _Template:
    def render(self, context):
        return ""
class _Env:
    def __init__(self):
        self.globals = {}
    def get_template(self, name):
        return _Template()
jinja2.Environment = lambda **kw: _Env()
jinja2.FileSystemLoader = lambda *a, **k: None
jinja2.pass_context = lambda f: f
sys.modules['jinja2'] = jinja2

# stub custom helper modules used by main
stub_db = types.ModuleType('db')
stub_db.chat_find_one = lambda *a, **kw: None
stub_db.chat_upsert = lambda *a, **kw: None
stub_db.resumes_all = lambda: []
stub_db.resumes_by_ids = lambda ids: []
stub_db.resumes_collection = None
stub_db.add_project_history = lambda *a, **kw: None
sys.modules['db'] = stub_db

stub_mongo_utils = types.ModuleType('mongo_utils')
stub_mongo_utils.update_resume = lambda *a, **kw: None
stub_mongo_utils.delete_resume_by_id = lambda *a, **kw: None
stub_mongo_utils.db = {"users": {}}
class _Users:
    def create_index(self, *a, **kw):
        pass
    def count_documents(self, *a, **kw):
        return 0
    def insert_one(self, *a, **kw):
        pass
    def find_one(self, *a, **kw):
        return None
    def delete_one(self, *a, **kw):
        return types.SimpleNamespace(deleted_count=0)
    def update_one(self, *a, **kw):
        return types.SimpleNamespace(modified_count=0)
    def find(self, *a, **kw):
        return []
stub_mongo_utils._users = _Users()
stub_mongo_utils.ENV = 'test'
stub_mongo_utils._guard = lambda op: False
sys.modules['mongo_utils'] = stub_mongo_utils

stub_pinecone_utils = types.ModuleType('pinecone_utils')
stub_pinecone_utils.add_resume_to_pinecone = lambda *a, **kw: None
stub_pinecone_utils.embed_text = lambda *a, **kw: []
stub_pinecone_utils.index = None
stub_pinecone_utils.search_best_resumes = lambda *a, **kw: []
sys.modules['pinecone_utils'] = stub_pinecone_utils
