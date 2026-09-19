import unittest, importlib.util
from pathlib import Path

class GateTest(unittest.TestCase):
 def test_fail_closed_and_role_mapping(self):
  path=Path(__file__).with_name('runner.py')
  self.assertTrue(path.exists(), 'repair runner missing')
  spec=importlib.util.spec_from_file_location('runner',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  v={k:True for k in m.CHECKS}
  v.update(A_matches_box='no',B_matches_box='yes',A_has_match_anywhere='no',B_has_match_anywhere='yes',A_unique='no',B_unique='yes',reason='visible evidence')
  self.assertTrue(m.accept(v,'B'))
  self.assertFalse(m.accept(v,'A'))
  for k in m.CHECKS:
   bad=dict(v); bad[k]='true'; self.assertFalse(m.accept(bad,'B'))
  bad=dict(v); bad['A_has_match_anywhere']='uncertain'; self.assertFalse(m.accept(bad,'B'))
  self.assertFalse(m.accept({},'A'))
  bad=dict(v); bad.pop('A_unique'); self.assertFalse(m.accept(bad,'B'))
  r={'set':'1996','sid':'x','ht':'object','img':'i','bbox':[0,0,1,1],'pos':'a','neg':'b'}
  self.assertEqual(m.content_id(r),m.content_id(dict(reversed(list(r.items())))))
  self.assertNotEqual(m.content_id(r),m.content_id({**r,'neg':'c'}))
  self.assertNotEqual(m.content_id(r),m.content_id({**r,'ht':'attribute'}))

if __name__=='__main__':unittest.main()
