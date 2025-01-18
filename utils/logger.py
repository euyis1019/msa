import datetime
import inspect
import os
import sys

class Logger:
    def __init__(self, log_dir=None, Name="console"):
        # 如果没有指定存储路径，默认为当前执行路径
        self.log_dir = log_dir if log_dir else os.getcwd()
        self.name = Name

    def log_var_shapes(self, **kwargs):
        caller_frame = inspect.stack()[1]
        caller_file = caller_frame.filename
        caller_line = caller_frame.lineno

        # 改变时间戳格式
        timestamp = datetime.datetime.now().strftime("%Y年%m月%d日_%H时%M分%S秒")
        log_file = f"tensor_shapes_{timestamp}.txt"
        log_file = os.path.join(self.log_dir, log_file)
        
        with open(log_file, "w") as file:
            file.write(f"Tensor shapes logged from: {caller_file}, line {caller_line}\n\n")
            for var_name, var_value in kwargs.items():
                if hasattr(var_value, 'shape'):
                    var_shape = var_value.shape
                    file.write(f"{var_name}: {var_shape}\n")
                elif hasattr(var_value, 'size'):
                    var_shape = var_value.size()
                    file.write(f"{var_name}: {var_shape}\n")
                else:
                    file.write(f"{var_name}: Not a tensor, the value is {var_value}\n")
        
        print(f"Tensor shapes logged to: {os.path.abspath(log_file)}")

    def save_console_output(self):
        class ConsoleLogger:
            def __init__(self, log_filename):
                self.terminal = sys.stdout
                # 确保文件夹存在，不存在则创建
                os.makedirs(os.path.dirname(log_filename), exist_ok=True)
                self.log = open(log_filename, "w")

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)

            def flush(self):
                self.terminal.flush()
                self.log.flush()

        # 改变时间戳格式，去掉“年”字，使用其他分隔符
        

        timestamp = datetime.datetime.now().strftime("%m月%d日_%H时%M分%S秒")
        log_file = f"{self.name}_{timestamp}.log"
        log_file = os.path.join(self.log_dir, log_file)
        sys.stdout = ConsoleLogger(log_file)
        sys.stderr = sys.stdout

        print(f"Console output is being logged to: {os.path.abspath(log_file)}")
if __name__ == "__main__":
    # 创建Logger实例并启用控制台输出重定向
    logger = Logger()
    logger.save_console_output()

    # 示例控制台输出
    print("This is a test message.")
