from visdom import Visdom
#连接visdom（http://127.0.0.1:8097/）
viz = Visdom()
viz.text('Hello World!')
viz.close()