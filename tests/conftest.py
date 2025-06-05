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
    'itsdangerous': {},
}
for name, attrs in mods.items():
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

# minimal jinja2 stub for TemplateResponse
if 'jinja2' not in sys.modules:
    jinja2 = types.ModuleType('jinja2')

    def pass_context(fn):
        return fn

    class Environment:
        def __init__(self, **kw):
            self.loader = kw.get('loader')
            self.globals = {}

        def get_template(self, name):
            class Tmpl:
                def render(self, ctx):
                    return ''
            return Tmpl()

    class FileSystemLoader:
        def __init__(self, *a, **k):
            pass

    jinja2.Environment = Environment
    jinja2.FileSystemLoader = FileSystemLoader
    jinja2.pass_context = pass_context
    jinja2.contextfunction = pass_context

    sys.modules['jinja2'] = jinja2

# simple itsdangerous stub
itsdangerous = types.ModuleType('itsdangerous')

class DummySigner:
    def dumps(self, obj):
        return 'token'

    def loads(self, token, max_age=None):
        return {'u': 'user', 'r': 'user'}

itsdangerous.URLSafeTimedSerializer = lambda *a, **k: DummySigner()
itsdangerous.BadSignature = Exception
sys.modules['itsdangerous'] = itsdangerous

# stub python-multipart import used by FastAPI
if 'multipart' not in sys.modules:
    multipart = types.ModuleType('multipart')
    multipart.__version__ = '0'
    sub = types.ModuleType('multipart.multipart')

    def parse_options_header(value):
        if not value:
            return b'', {}
        parts = value.split(';')
        main = parts[0].strip().encode()
        params = {}
        for part in parts[1:]:
            if '=' in part:
                k, v = part.strip().split('=', 1)
                params[k.encode()] = v.strip('"')
        return main, params

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
            if b"\r\n\r\n" not in data:
                return FormData([])
            header, body = data.split(b"\r\n\r\n", 1)
            body = body.strip()
            header_str = header.decode()
            name = 'file'
            if 'name="' in header_str:
                name = header_str.split('name="')[1].split('"')[0]
            filename = 'upload'
            if 'filename="' in header_str:
                filename = header_str.split('filename="')[1].split('"')[0]
            return FormData([(name, UploadFile(filename=filename, file=io.BytesIO(body)))])

    class MultiPartException(Exception):
        def __init__(self, message):
            self.message = message

    sub.parse_options_header = parse_options_header
    sys.modules['multipart.multipart'] = sub
    multipart.parse_options_header = parse_options_header
    multipart.QuerystringParser = QuerystringParser
    multipart.MultiPartParser = MultiPartParser
    multipart.MultiPartException = MultiPartException
    sys.modules['multipart'] = multipart

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
            base = kwargs.get('base_url', '')
            self.base_url = types.SimpleNamespace(join=lambda url, b=base: b + url)

        def request(self, method, url, **kwargs):
            req = Request(method, url)
            data = kwargs.get('data')
            if isinstance(data, dict):
                from urllib.parse import urlencode
                req.headers['content-type'] = 'application/x-www-form-urlencoded'
                req._content = urlencode(data).encode()
            files = kwargs.get('files')
            if isinstance(files, dict):
                boundary = 'BOUNDARY'
                parts = []
                for name, (filename, content, ctype) in files.items():
                    parts.append(
                        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n".encode() + content + b"\r\n"
                    )
                parts.append(f"--{boundary}--\r\n".encode())
                req.headers['content-type'] = f'multipart/form-data; boundary={boundary}'
                body = b''.join(parts)
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
stub_mongo_utils._users = {}
stub_mongo_utils.ENV = 'test'
stub_mongo_utils._guard = lambda op: False
sys.modules['mongo_utils'] = stub_mongo_utils

stub_pinecone_utils = types.ModuleType('pinecone_utils')
stub_pinecone_utils.add_resume_to_pinecone = lambda *a, **kw: None
stub_pinecone_utils.embed_text = lambda *a, **kw: []
stub_pinecone_utils.index = None
stub_pinecone_utils.search_best_resumes = lambda *a, **kw: []
sys.modules['pinecone_utils'] = stub_pinecone_utils
