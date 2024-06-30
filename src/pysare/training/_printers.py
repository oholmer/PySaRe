
import time

class Printer():
    def __init__(self, print_time) -> None:

        self.print_time = print_time

        self.last_print = time.time()-self.print_time

        self.last_train_loss = f"{'-':>12}"
        self.last_validation_loss = f"{'-':>12}"

    def finalize(self):
        print("")
        print("-"*57)

    def loss_str(self, loss):
        if loss:
            return f"{loss:>12.6g}"
        else:
            return f"{'-':>12}"

    def int_str(self, num):
        if num < 1000000:
            return f"{num:6.0f}"
        else:
            return f"{num:>2.2g}"

    def train_update(self, epoch, batch, batch_loss):
        self.batch_loss = self.loss_str(batch_loss)
        self.epoch = self.int_str(epoch)
        self.batch = self.int_str(batch)

        self.print()
    
    def print(self):
        pass

class SingleLinePrinter(Printer):
    def __init__(self, print_time) -> None:
        super().__init__(print_time)

        print(f"{'Current batch:':<26} | {'Previous epoch:':<27}")
        print(f"{'Epoch':>6} {'Batch':>6} {'Train. Loss':>12} | {'Train. loss':>12}{'':>3}{'Valid. loss':>12}")
        print("-"*57)

    def finalize_update(self, epoch, batch, train_loss, validation_loss, batch_loss):
        self.last_train_loss = self.loss_str(train_loss)
        self.last_validation_loss = self.loss_str(validation_loss)
        self.batch_loss = self.loss_str(batch_loss)
        self.epoch = self.int_str(epoch)
        self.batch = self.int_str(batch)

        self.print()

    def print(self):
        if time.time()-self.last_print > self.print_time:
            print(self.epoch + " " + self.batch + " " + self.batch_loss + " | " +
                  self.last_train_loss + "   "+self.last_validation_loss, end='\r')
            self.last_print = time.time()


class MultipleLinePrinter(Printer):

    def __init__(self, print_time, print_new_line_count=1) -> None:
        super().__init__(print_time)

        self.print_new_line_count = print_new_line_count
        self.skip_count = 0

        print(f"{'Current batch:':<26} | {'Full epoch:':<27}")
        print(f"{'Epoch':>6} {'Batch':>6} {'Train. Loss':>12} | {'Train. loss':>12}{'':>3}{'Valid. loss':>12}")
        print("-"*57)

        self.last_train_loss = f"{'-':>12}"
        self.last_validation_loss = f"{'-':>12}"
        self.print_finalize = False

    def finalize_update(self, epoch, batch, train_loss, validation_loss, batch_loss):
        self.last_train_loss = self.loss_str(train_loss)
        self.last_validation_loss = self.loss_str(validation_loss)
        self.batch_loss = self.loss_str(batch_loss)
        self.epoch = self.int_str(epoch)
        self.batch = self.int_str(batch)
        self.print(finalize=True)
        self.last_train_loss = f"{'-':>12}"
        self.last_validation_loss = f"{'-':>12}"

    def _print(self, end='\r'):
        print(self.epoch + " " + self.batch + " " + self.batch_loss + " | " +
              self.last_train_loss + "   "+self.last_validation_loss, end=end)

    def print(self, finalize=False):
        if finalize:

            self.skip_count += 1

            if self.skip_count >= self.print_new_line_count:
                if self.print_finalize or (time.time()-self.last_print > self.print_time):
                    self._print("\n")
                    self.skip_count = 0
                    self.last_print = time.time()
                    self.print_finalize = False
                    return
            self.print_finalize = False
        if time.time()-self.last_print > self.print_time:
            self._print()
            self.last_print = time.time()
            self.print_finalize = True

