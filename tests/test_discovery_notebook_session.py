"""실제 requests/google-auth 타입으로 notebook factory를 실행한다. 네트워크 없음."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def available():
    try:return importlib.util.find_spec('google.auth') is not None
    except ModuleNotFoundError:return False


@unittest.skipUnless(available(),'보존 transport_deps의 google-auth/requests 환경 필요')
class NotebookSessionTests(unittest.TestCase):
    def environment(self):
        import requests
        from google.auth.transport.requests import AuthorizedSession,Request
        from google.auth.credentials import AnonymousCredentials
        from urllib3.util.retry import Retry
        from rewind_archive.request_policy import validate_no_hidden_retries
        return dict(requests=requests,AuthorizedSession=AuthorizedSession,Request=Request,Retry=Retry,
                    credentials=AnonymousCredentials(),validate_no_hidden_retries=validate_no_hidden_retries)

    def factory(self,env):
        nb=json.loads((ROOT/'notebooks/archvis_rgb_discovery.ipynb').read_bytes())
        tree=ast.parse(''.join(nb['cells'][3]['source']))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='session_factory')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_notebook_factory','exec'),env)
        return env['session_factory']

    def test_old_outer_only_factory_fails_on_real_hidden_auth_retry(self):
        env=self.environment()
        # 677bc7f notebook 원형: 외부 adapter만0이어도 내부 auth retry3이 남는다.
        code='''def old_factory():
    session = AuthorizedSession(credentials)
    session._max_refresh_attempts = 0
    for adapter in session.adapters.values():
        adapter.max_retries = Retry(total=0, connect=0, read=0, redirect=0, status=0)
    return session
'''
        exec(code,env)
        with patch('requests.sessions.Session.request',side_effect=AssertionError('no network')) as request, \
             patch('socket.create_connection',side_effect=AssertionError('no network')) as socket:
            session=env['old_factory']()
            try:
                with self.assertRaisesRegex(Exception,'session_retry_policy'):env['validate_no_hidden_retries'](session)
                self.assertTrue(any(a.max_retries.total!=0 for a in session._auth_request.session.adapters.values()))
            finally:session.close()
            request.assert_not_called();socket.assert_not_called()

    def test_actual_factory_validates_before_return_and_closes_both_real_sessions(self):
        import requests
        from google.auth.transport.requests import AuthorizedSession,Request
        env=self.environment();checked=[];original=env['validate_no_hidden_retries']
        def validated(session):checked.append(session);return original(session)
        env['validate_no_hidden_retries']=validated;factory=self.factory(env)
        closed=[];real_close=requests.Session.close
        def close(session):closed.append(session);return real_close(session)
        with patch('requests.sessions.Session.request',side_effect=AssertionError('no network')) as request, \
             patch('socket.create_connection',side_effect=AssertionError('no network')) as socket, \
             patch.object(requests.Session,'close',autospec=True,side_effect=close):
            session=factory();auth_http=session._auth_request.session
            self.assertIsInstance(session,AuthorizedSession);self.assertIsInstance(session._auth_request,Request)
            self.assertIsInstance(auth_http,requests.Session);self.assertEqual(checked,[session])
            self.assertFalse(session.trust_env);self.assertFalse(auth_http.trust_env)
            self.assertEqual(session._max_refresh_attempts,0)
            self.assertTrue(all(a.max_retries.total==0 for s in (session,auth_http) for a in s.adapters.values()))
            session.close();self.assertEqual(closed,[session,auth_http])
            request.assert_not_called();socket.assert_not_called()

    def test_preflight_rejection_closes_both_sessions_without_any_request(self):
        import requests
        env=self.environment();closed=[];real_close=requests.Session.close
        def fail(session):raise ValueError('preflight_rejected')
        def close(session):closed.append(session);return real_close(session)
        env['validate_no_hidden_retries']=fail;factory=self.factory(env)
        with patch('requests.sessions.Session.request',side_effect=AssertionError('no network')) as request, \
             patch.object(requests.Session,'close',autospec=True,side_effect=close):
            with self.assertRaisesRegex(ValueError,'preflight_rejected'):factory()
            self.assertEqual(len(closed),2);request.assert_not_called()

    def test_authorized_constructor_failure_closes_auth_transport(self):
        import requests
        env=self.environment();closed=[];real_close=requests.Session.close
        def fail(*a,**k):raise ValueError('constructor_rejected')
        def close(session):closed.append(session);return real_close(session)
        env['AuthorizedSession']=fail;factory=self.factory(env)
        with patch('requests.sessions.Session.request',side_effect=AssertionError('no network')) as request, \
             patch.object(requests.Session,'close',autospec=True,side_effect=close):
            with self.assertRaisesRegex(ValueError,'constructor_rejected'):factory()
            # Request 소멸자도 같은 auth session을 닫을 수 있다. 새 외부 session은 없다.
            self.assertTrue(closed);self.assertEqual(len({id(s) for s in closed}),1)
            request.assert_not_called()
