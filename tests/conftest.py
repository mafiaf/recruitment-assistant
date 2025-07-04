import sys
import types

# ── heavyweight deps we want to stub ──────────────────────────────────────────
mods = {
    'markdown': {'markdown': lambda text, extensions=None: text},
    'mammoth': {},
    'numpy': {'array': lambda x: x},
    'openai': {},
    'pdfplumber': {},
    'PyPDF2': {'PdfReader': lambda *a, **k: types.SimpleNamespace(pages=[])},
    'docx': {'Document': lambda *a, **k: None},
    'dotenv': {'load_dotenv': lambda *a, **k: None},
    'certifi': {'where': lambda: ''},
    'bleach': {'clean': lambda text, tags=None, strip=False: text},
    # itsdangerous minimal signer
    'itsdangerous': {
        'URLSafeTimedSerializer': lambda *a, **k: types.SimpleNamespace(
            dumps=lambda obj: 'token',
            loads=lambda t: {},
        ),
        'BadSignature': Exception,
    },
    # jinja2 stub sufficient for Starlette / FastAPI
    'jinja2': {
        'Environment': type(
            'Env',
            (),
            {
                '__init__': lambda self, **kw: None,
                'globals': {},
                'get_template': lambda self, name: types.SimpleNamespace(
                    render=lambda ctx: ''
                ),
            },
        ),
        'FileSystemLoader': lambda *a, **k: None,
        'select_autoescape': lambda *a, **k: None,
        'Template': types.SimpleNamespace,
        'pass_context': lambda f: f,
        'contextfunction': lambda f: f,
    },
    # top-level multipart package with placeholder submodule
    'multipart': {
        '__version__': '0',
        'multipart': types.ModuleType('multipart.multipart'),
    },
}

for name, attrs in mods.items():
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        if name == 'multipart':
            sys.modules['multipart.multipart'] = attrs['multipart']

# ── enrich multipart for form / file handling ────────────────────────────────
from urllib.parse import parse_qs

if 'multipart' in sys.modules:
    def parse_options_header(value):
        if not value:
            return b'', {}
        main, *params = value.split(';')
        parsed = {}
        for p in params:
            if '=' in p:
                k, v = p.strip().split('=', 1)
                parsed[k.encode()] = v.strip('"')
        return main.strip().encode(), parsed

    class QuerystringParser:
        def __init__(self, callbacks):
            self.callbacks = callbacks
            self.buf = b''

        def write(self, data):
            self.buf += data

        def finalize(self):
            pairs = self.buf.split(b'&') if self.buf else []
            for p in pairs:
                if not p:
                    continue
                name, _, value = p.partition(b'=')
                self.callbacks['on_field_start']()
                self.callbacks['on_field_name'](name, 0, len(name))
                self.callbacks['on_field_data'](value, 0, len(value))
                self.callbacks['on_field_end']()
            self.callbacks['on_end']()

    class MultiPartParser:
        def __init__(self, headers, stream, **kw):
            self.headers = headers
            self.stream = stream

        async def parse(self):
            import io
            from starlette.datastructures import UploadFile, FormData
            data = b''
            async for chunk in self.stream:
                data += chunk
            if b'\r\n\r\n' not in data:
                return FormData([])
            header, body = data.split(b'\r\n\r\n', 1)
            header_str = header.decode()
            name = 'file'
            if 'name="' in header_str:
                name = header_str.split('name="')[1].split('"')[0]
            filename = 'upload'
            if 'filename="' in header_str:
                filename = header_str.split('filename="')[1].split('"')[0]
            return FormData(
                [(name, UploadFile(filename=filename, file=io.BytesIO(body.strip())))]
            )

    class MultiPartException(Exception):
        pass

    m = sys.modules['multipart']
    m.parse_options_header = parse_options_header
    m.QuerystringParser = QuerystringParser
    m.MultiPartParser = MultiPartParser
    m.MultiPartException = MultiPartException
    sys.modules['multipart.multipart'].parse_options_header = parse_options_header

if 'multipart.multipart' not in sys.modules:
    sub = types.ModuleType('multipart.multipart')
    sub.parse_options_header = lambda *_a, **_k: (b'', {})
    sys.modules['multipart.multipart'] = sub
    if 'multipart' in sys.modules:
        sys.modules['multipart'].multipart = sub

# ── httpx stub with form / file support ───────────────────────────────────────
if 'httpx' not in sys.modules:
    httpx = types.ModuleType('httpx')

    class Request:
        def __init__(self, method, url, data=None):
            from urllib.parse import urlsplit, urlencode
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
            if isinstance(data, bytes):
                self._content = data
            elif isinstance(data, str):
                self._content = data.encode()
            elif data is not None:
                self._content = urlencode(data, doseq=True).encode()
            else:
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

    class _URL(str):
        def join(self, url: str):
            if url.startswith('http'):
                return url
            if not url.startswith('/'):
                url = '/' + url
            return self.rstrip('/') + url

    class Client:
        def __init__(self, *args, **kwargs):
            self._transport = kwargs.get('transport')
            self.base_url = _URL(kwargs.get('base_url', ''))

        def request(self, method, url, **kwargs):
            req = Request(method, url)
            data = kwargs.get('data')
            files = kwargs.get('files')

            if isinstance(data, dict):
                from urllib.parse import urlencode
                req.headers['content-type'] = 'application/x-www-form-urlencoded'
                req._content = urlencode(data).encode()

            if isinstance(files, dict):
                boundary = 'BOUNDARY'
                parts = []
                for name, (filename, content, ctype) in files.items():
                    parts.append(
                        (
                            f'--{boundary}\r\n'
                            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                            f'Content-Type: {ctype}\r\n\r\n'
                        ).encode()
                        + content
                        + b'\r\n'
                    )
                parts.append(f'--{boundary}--\r\n'.encode())
                body = b''.join(parts)
                req.headers['content-type'] = f'multipart/form-data; boundary={boundary}'
                req.headers['content-length'] = str(len(body))
                req._content = body

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
        UseClientDefault=object(),
        TimeoutTypes=object(),
    )
    httpx._types = types.SimpleNamespace(
        URLTypes=object(),
        HeaderTypes=object(),
        QueryParamTypes=object(),
        CookieTypes=object(),
        RequestContent=object(),
        RequestFiles=object(),
        AuthTypes=object(),
    )
    sys.modules['httpx'] = httpx

# ── starlette templating shim ─────────────────────────────────────────────────
if 'starlette.templating' not in sys.modules:
    st_templates = types.ModuleType('starlette.templating')

    class DummyTemplates:
        def __init__(self, *a, **kw):
            pass

        def TemplateResponse(self, name, context, status_code=200, **kw):
            from starlette.responses import HTMLResponse
            return HTMLResponse('', status_code=status_code)

        def get_template(self, name):
            return types.SimpleNamespace(render=lambda **kw: '')

    st_templates.Jinja2Templates = DummyTemplates
    sys.modules['starlette.templating'] = st_templates

    fastapi_templating = types.ModuleType('fastapi.templating')
    fastapi_templating.Jinja2Templates = DummyTemplates
    sys.modules['fastapi.templating'] = fastapi_templating

# ── Starlette form parser fallback ────────────────────────────────────────────
import starlette.formparsers as fp
from starlette.datastructures import FormData

class SimpleFormParser:
    def __init__(self, headers, stream):
        self.headers = headers
        self.stream = stream

    async def parse(self):
        body = b''
        async for chunk in self.stream:
            body += chunk
        data = {
            k: v[0] if len(v) == 1 else v
            for k, v in parse_qs(body.decode()).items()
        }
        return FormData(data)

fp.FormParser = SimpleFormParser

# ── passlib stubs ─────────────────────────────────────────────────────────────
passlib_context = types.ModuleType('passlib.context')
passlib_context.CryptContext = lambda **kw: types.SimpleNamespace(
    hash=lambda p: p,
    verify=lambda p, h: p == h,
)
sys.modules['passlib.context'] = passlib_context

passlib_hash = types.ModuleType('passlib.hash')
passlib_hash.bcrypt = types.SimpleNamespace(
    hash=lambda p: p,
    verify=lambda p, h: p == h,
)
sys.modules['passlib.hash'] = passlib_hash

# ── Mongo, bson plus helpers ──────────────────────────────────────────────────
pymongo = types.ModuleType('pymongo')
pymongo.MongoClient = lambda *a, **k: None
pymongo.errors = types.SimpleNamespace(PyMongoError=Exception)
sys.modules['pymongo'] = pymongo

# async Mongo driver stub
motor = types.ModuleType('motor')
motor_asyncio = types.ModuleType('motor.motor_asyncio')

class DummyMotorClient:
    def __init__(self, *a, **k):
        self.admin = types.SimpleNamespace(command=lambda *a, **k: None)

    def __getitem__(self, name):
        return DummyDB()

motor_asyncio.AsyncIOMotorClient = DummyMotorClient
motor.motor_asyncio = motor_asyncio
sys.modules['motor'] = motor
sys.modules['motor.motor_asyncio'] = motor_asyncio

bson = types.ModuleType('bson')
bson.ObjectId = str
sys.modules['bson'] = bson

# ── db stub -------------------------------------------------------------------
stub_db = types.ModuleType('db')
stub_db.chat_find_one = lambda *a, **kw: None
stub_db.chat_upsert = lambda *a, **kw: None
stub_db.resumes_all = lambda: []
stub_db.resumes_by_ids = lambda ids: []
stub_db.resumes_collection = None
stub_db.add_project_history = lambda *a, **kw: None
sys.modules['db'] = stub_db

# ── mongo_utils merged stub ---------------------------------------------------
stub_mongo_utils = types.ModuleType('mongo_utils')
async def _async_none(*a, **kw):
    return None

async def _async_list(*a, **kw):
    return []

async def _async_one(*a, **kw):
    return 1

stub_mongo_utils.update_resume = _async_one
stub_mongo_utils.delete_resume_by_id = _async_one
stub_mongo_utils.chat_find_one = _async_none
stub_mongo_utils.chat_upsert = _async_none
stub_mongo_utils.resumes_all = _async_list
stub_mongo_utils.resumes_by_ids = _async_list
stub_mongo_utils.resumes_page = _async_list
stub_mongo_utils.resumes_count = lambda *a, **kw: 0
stub_mongo_utils.resumes_collection = None
stub_mongo_utils.add_project_history = _async_none
stub_mongo_utils.delete_project = _async_one
stub_mongo_utils.update_project_description = _async_one
stub_mongo_utils.update_project_status = _async_one
stub_mongo_utils.ensure_indexes = _async_none
class _Chats:
    async def update_one(self, *a, **kw):
        return None

stub_mongo_utils.chats = _Chats()

class DummyColl:
    def __getattr__(self, name):
        return lambda *a, **k: None

class DummyDB(dict):
    def __getitem__(self, key):
        return DummyColl()

stub_mongo_utils.db = DummyDB()

class _Users:
    async def create_index(self, *a, **kw):
        pass

    async def count_documents(self, *a, **kw):
        return 0

    async def insert_one(self, *a, **kw):
        pass

    async def find_one(self, *a, **kw):
        return None

    async def delete_one(self, *a, **kw):
        return types.SimpleNamespace(deleted_count=0)

    async def update_one(self, *a, **kw):
        return types.SimpleNamespace(modified_count=0)

    async def find(self, *a, **kw):
        return []

stub_mongo_utils._users = _Users()
stub_mongo_utils.ENV = 'test'
stub_mongo_utils._guard = lambda op: False
sys.modules['mongo_utils'] = stub_mongo_utils

# ── FastAPI TestClient stub ───────────────────────────────────────────────────
if 'fastapi.testclient' not in sys.modules:
    testclient = types.ModuleType('fastapi.testclient')

    class TestClient:
        def __init__(self, app):
            self.app = app

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def get(self, path, **kw):
            import main, types as _t
            if path == '/login':
                req = _t.SimpleNamespace(url=_t.SimpleNamespace(path=path), cookies={})
                return main.login_form(req)
            return _t.SimpleNamespace(status_code=404)

        def post(self, path, data=None, files=None, **kw):
            import main, asyncio, types as _t
            if path == '/login':
                req = _t.SimpleNamespace(url=_t.SimpleNamespace(path=path), cookies={})
                return asyncio.run(
                    main.login_post(
                        req,
                        username=data.get('username'),
                        password=data.get('password'),
                    )
                )
            if path == '/upload_resume':
                req = _t.SimpleNamespace(url=_t.SimpleNamespace(path=path), cookies={})
                file = None
                if files and 'file' in files:
                    filename, content = files['file'][:2]
                    async def _read():
                        return content
                    file = _t.SimpleNamespace(filename=filename, read=_read)
                return asyncio.run(
                    main.upload_resume(
                        req,
                        file=file,
                        name=(data or {}).get('name'),
                        text=(data or {}).get('text'),
                        resume=None,
                    )
                )
            return _t.SimpleNamespace(status_code=404)

    testclient.TestClient = TestClient
    sys.modules['fastapi.testclient'] = testclient

# ── pinecone utils stub ───────────────────────────────────────────────────────
stub_pinecone_utils = types.ModuleType('pinecone_utils')
stub_pinecone_utils.add_resume_to_pinecone = lambda *a, **kw: None
stub_pinecone_utils.embed_text = lambda *a, **kw: []
stub_pinecone_utils.index = None
stub_pinecone_utils.search_best_resumes = lambda *a, **kw: []
stub_pinecone_utils.add_project_to_pinecone = lambda *a, **kw: None
stub_pinecone_utils.search_best_projects = lambda *a, **kw: []
stub_pinecone_utils.delete_project_from_pinecone = lambda *a, **kw: None
sys.modules['pinecone_utils'] = stub_pinecone_utils

# ── finally load main.py so tests can import it ───────────────────────────────
import importlib.util
import pathlib

root_dir = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir))
main_path = root_dir / 'main.py'
spec = importlib.util.spec_from_file_location('main', main_path)
main = importlib.util.module_from_spec(spec)
sys.modules['main'] = main
spec.loader.exec_module(main)
