import sys
sys.path.insert(0, sys.argv[1])
import vsight_reward_plugin_group as m
cases=[
 ('{"binding":"MATCH","decision":"KEEP"}','KEEP','relation','ROH'),
 ('{"binding":"NA","decision":"REJECT"}','REJECT','object','BOH'),
 ('bad','KEEP','relation','ROH'),
 ('{"binding":"MISMATCH","decision":"KEEP"}','REJECT','relation','ROH'),
]
for args in cases:
 print(args, '=>', m.one(*args))
assert m.one(*cases[0]) == (0.1,1.5,1.5)
assert m.one(*cases[1]) == (0.1,0.0,1.0)
assert m.one(*cases[2]) == (0.0,0.0,-0.2)
assert m.one(*cases[3]) == (0.1,1.5,0.0)
print('registered', sorted(k for k in m.orms if k.startswith('vsight_group')))
