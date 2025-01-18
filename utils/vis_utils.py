try:
    import visdom
except ImportError:
    print("请先使用 pip install visdom 安装visdom包")
    raise

class VisdomLogger:
    def __init__(self, env_name='main'):
        self.vis = visdom.Visdom(env=env_name)
        self.best_test_accuracy = 0
        self.best_epoch = 0

        # 创建窗口
        self.vis.line(
            X=[0],
            Y=[[0, 0]],
            win='test_accuracy_over_time',
            opts=dict(
                xlabel='Epoch',
                ylabel='Accuracy',
                title='Test Accuracy Over Epochs',
                legend=['Test Accuracy', 'Best Test Accuracy']
            )
        )

    def close_all(self):
        self.vis.close(win='train_loss')
        self.vis.close(win='train_accuracy')
        self.vis.close(win='test_loss')
        self.vis.close(win='test_accuracy')

    def log_metrics(self, epoch, train_loss, train_accuracy, test_loss, test_accuracy):
        self.vis.line(X=[epoch], Y=[train_loss], win='train_loss', update='append', opts=dict(title='Train Loss'))
        self.vis.line(X=[epoch], Y=[train_accuracy], win='train_accuracy', update='append', opts=dict(title='Train Accuracy'))
        self.vis.line(X=[epoch], Y=[test_loss], win='test_loss', update='append', opts=dict(title='Test Loss'))
        self.vis.line(X=[epoch], Y=[test_accuracy], win='test_accuracy', update='append', opts=dict(title='Test Accuracy'))

        # 更新最佳测试精度
        if test_accuracy > self.best_test_accuracy:
            self.best_test_accuracy = test_accuracy
            self.best_epoch = epoch

        # 更新测试精度图表
        self.vis.line(
            X=[epoch],
            Y=[[test_accuracy, self.best_test_accuracy]],
            win='test_accuracy_over_time',
            update='append'
        )

    def log_best_metrics(self):
        # 在最佳点处添加标注
        self.vis.text(
            f'Best Epoch: {self.best_epoch}, Best Test Accuracy: {self.best_test_accuracy:.2f}%',
            win='best_test_accuracy',
            opts=dict(
                title='Best Test Accuracy'
            )
        )
