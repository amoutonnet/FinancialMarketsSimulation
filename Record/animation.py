def init_animation(self):
        self.create_figure()
        self.animation_started = True
        self.animation_fig.canvas.draw()
        plt.show(block=False)


def step_animation(self):
    x = np.arange(self.max_steps - self.window_size, self.max_steps, 1)
    if self.max_steps == self.window_size:
        for id_mm in self.market_makers:
            for i, j in [(0, 'ask_price'), (2, 'portfolio')]:
                self.axes[i].plot(x, self.window_data[self.data_idx['%s_%d' % (j, id_mm)]][:-1], label=self.market_makers[id_mm].short_name, alpha=0.8)
        for id_d in self.dealers:
            for i, j in [(1, 'cash_dealer'), (3, 'portfolio_value')]:
                self.axes[i].plot(x, self.window_data[self.data_idx['%s_%d' % (j, id_d)]][:-1], alpha=0.8)
            self.axes[4].plot(x, self.window_data[self.data_idx['cash_dealer_%d' % id_d]][:-1] + self.window_data[self.data_idx['portfolio_value_%d' % id_d]][:-1], alpha=0.8)
        self.axes[0].legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, 1.22), prop={'size': 9})
        self.animation_fig.tight_layout()
        for i in range(len(self.lines)):
            self.lines[i] = self.axes[i].get_lines()
    else:
        for id_mm in self.market_makers:
            for i, j in [(0, 'ask_price'), (2, 'portfolio')]:
                self.lines[i][id_mm].set_data(x, self.window_data[self.data_idx['%s_%d' % (j, id_mm)]][:-1])
                self.axes[i].draw_artist(self.lines[i][id_mm])
        for id_d in self.dealers:
            for i, j in [(1, 'cash_dealer'), (3, 'portfolio_value')]:
                self.lines[i][id_d].set_data(x, self.window_data[self.data_idx['%s_%d' % (j, id_d)]][:-1])
                self.axes[i].draw_artist(self.lines[i][id_d])
            self.lines[4][id_d].set_data(x, self.window_data[self.data_idx['cash_dealer_%d' % id_d]][:-1] + self.window_data[self.data_idx['portfolio_value_%d' % id_d]][:-1])
            self.axes[4].draw_artist(self.lines[i][id_d])
        for ax in self.axes:
            ax.relim()
            ax.autoscale_view()
        self.animation_fig.canvas.draw()
        self.animation_fig.canvas.flush_events()
