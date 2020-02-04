import neologdn

print(neologdn.normalize('「初めてのTensorFlow」は定価2200円+税です'))
print(neologdn.normalize('「初めての　ＴｅｎｓｏｒＦｌｏｗ」は定価２２００円＋税です'))
print(neologdn.normalize('｢初めての TensorFlow｣は定価2200円＋税です'))
