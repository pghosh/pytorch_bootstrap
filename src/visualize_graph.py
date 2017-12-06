import numpy as np

class Visualize_graph:
    def __init__(self):
        self.wins = {}

    def _update_line(self,x, y, viz, title, legend=None, opts=None):
        ''' Updates or creates a line plot and appends x, y point '''
        xx = np.array([x])
        yy = np.array([y])
        if not title in self.wins:
            if not opts:
                opts = {'title': title, 'markersize':1, 'legend':[legend]}
            win = viz.line(X=xx, Y=yy, opts=opts)
            self.wins[title] = win
        else:
            viz.updateTrace(X=xx, Y=yy, win=self.wins[title], name=legend)