import tkinter as tk
from core import *

class HighlightText(tk.Text):
    # Highlight the content with its corresponding statues

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def highlight_line(self, line_number, status):
        tag = f'line{line_number}'
        self.tag_remove(tag, f'{line_number}.0', f'{line_number}.end')

        if status == "proof":
            self.tag_configure(tag, background='green')
        elif status == "hold":
            self.tag_configure(tag, background='yellow')
        elif status == "error":
            self.tag_configure(tag, background='red')
        self.tag_add(tag, f'{line_number}.0', f'{line_number}.end')

class ChildWindow(tk.Frame):
    # Window for step management

    def __init__(self, master, step, informal, formal, main_window, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master
        self.step = step
        self.informal = informal
        self.formal = formal

        self.main_window = main_window

        self.status = None

        self.label = tk.Label(self, text=informal)
        self.label.pack(side=tk.TOP, fill=tk.X)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        self.proof_button = tk.Button(self.button_frame, text="PROOF", command=self.proof_check)
        self.proof_button.pack(side=tk.LEFT)

        self.check_button = tk.Button(self.button_frame, text="CHECK", command=self.check_formal)
        self.check_button.pack(side=tk.LEFT)
        self.action_button2_hide = True

        self.regen_button = tk.Button(self.button_frame, text="REGEN")
        self.regen_button.pack(side=tk.LEFT)

        self.hold_button = tk.Button(self.button_frame, text="HOLD", command=self.proof_hold)
        self.hold_button.pack(side=tk.LEFT)

        self.delete_button = tk.Button(self.button_frame, text="UNDO", command=self.delete_window)
        self.delete_button.pack(side=tk.LEFT)

    def proof_check(self):
        info = self.main_window.step_check()
        if info == 0:
            self.status = "proof"
            self.main_window.status_bar.config(text=f"Status: Proof has been completed.")
        elif info == 1:
            self.status = "proof"
            self.main_window.status_bar.config(text=f"Status: Current step has been verified. Please continue your proof.")
        else:
            self.status = "error"
            self.main_window.status_bar.config(text=f"Status: Proof failed at current step.")
        self.main_window.update_content()

    def proof_hold(self):
        self.status = "hold"
        self.main_window.update_content()
        self.main_window.status_bar.config(text=f"Status: Current step is placed on hold. Please continue your proof.")

    def lock(self):
        self.proof_button.config(state=tk.DISABLED)
        self.regen_button.config(state=tk.DISABLED)
        self.hold_button.config(state=tk.DISABLED)
        self.delete_button.config(state=tk.DISABLED)

    def unlock(self):
        self.proof_button.config(state=tk.NORMAL)
        self.regen_button.config(state=tk.NORMAL)
        self.hold_button.config(state=tk.NORMAL)
        self.delete_button.config(state=tk.NORMAL)

    def delete_window(self):
        self.main_window.remove_child_window(self)

    def check_formal(self):
        if self.action_button2_hide:
            self.label.config(text=f'{self.informal}\n{self.formal}')
            self.action_button2_hide = False
            self.check_button.config(text="HIDE")
        else:
            self.label.config(text=self.informal)
            self.action_button2_hide = True
            self.check_button.config(text="CHECK")


class MainWindow(tk.Tk):
    # The MainWindow for proof edition

    def __init__(self):
        super().__init__()

        self.title("Step Proof")
        self.geometry('800x500')

        self.steps = 0
        self.proofs = []
        self.child_windows = []

        self.informal_question = None
        self.informal_question = None
        self.informal_proofs = []
        self.formal_proofs = []

        self.model_name = "GPT3.5"

        self.central_frame = tk.Frame(self)
        self.central_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.central_frame)
        self.scrollbar = tk.Scrollbar(self.central_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.question_window = tk.Entry(self.scrollable_frame)
        self.question_window.pack(fill=tk.X)

        self.question_button = tk.Button(self.scrollable_frame, text="Upload Problem", command=self.add_problem)
        self.question_button.pack(fill=tk.X)

        self.content_window = HighlightText(self.scrollable_frame, height=20, width=100)
        self.content_window.pack(fill=tk.X)
        self.content_window.config(state=tk.DISABLED)

        self.proof_window = tk.Entry(self.scrollable_frame)
        self.proof_window.pack(fill=tk.X)
        self.proof_window.config(state=tk.DISABLED)

        self.add_proof_button = tk.Button(self.scrollable_frame, text="Add Proof", command=self.add_step_proof)
        self.add_proof_button.pack(fill=tk.X)
        self.add_proof_button.config(state=tk.DISABLED)

        self.create_menu_bar()

        self.status_bar = tk.Label(self, text="Status: Please connect to LLM and Checker", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_menu_bar(self):
        menu_bar = tk.Menu(self)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Exit", command=self.quit)
        file_menu.add_command(label="Reset", command=self.reset)
        file_menu.add_command(label="Connect", command=self.connect)
        menu_bar.add_cascade(label="File", menu=file_menu)

        self.config(menu=menu_bar)

    def connect(self):
        self.isabelle = Checker()
        self.model = LLM(self.model_name)
        self.status_bar.config(text=f"Status: Has connected to the {self.model_name}")

    def step_check(self):
        proof = f"{self.formal_question}\nproof-\n"
        for child_window in self.child_windows:
            if child_window.status == 'proof':
                formal = child_window.formal.split('sledgehammer')[0] + 'sorry'
                proof += f"(*{child_window.informal}*)\n{formal}\n"
            elif child_window.status == "hold":
                if 'sledgehammer' not in child_window.formal:
                    formal = child_window.formal.split('by')[0] + 'sorry'
                    proof += f"(*{child_window.informal}*)\n{formal}\n"
            else:
                proof += f"(*{child_window.informal}*)\n{child_window.formal}\n"
        proof += 'qed\n'
        print(proof)
        self.isabelle.upload(proof)
        info = self.isabelle.check()
        print(info)
        return info

    def reset(self):
        self.informal_question = None
        self.informal_proofs = []
        self.content_window.config(state=tk.NORMAL)
        self.content_window.delete('1.0', tk.END)
        self.content_window.config(state=tk.DISABLED)
        self.question_window.config(state=tk.NORMAL)
        self.question_window.delete(0, tk.END)
        self.question_button.config(state=tk.NORMAL)
        self.proof_window.config(state=tk.DISABLED)
        self.add_proof_button.config(state=tk.DISABLED)
        self.steps = 0
        self.proofs = []
        for child_window in self.child_windows:
            child_window.destroy()
        self.child_windows = []

    def add_problem(self):
        self.informal_question = self.question_window.get()
        self.formal_question = self.model.problem_formalization(self.informal_question)
        print(self.formal_question)
        question = f"PROBLEM:\n{self.informal_question}\n\n"

        self.content_window.config(state=tk.NORMAL)
        self.content_window.insert(tk.END, question)
        self.content_window.config(state=tk.DISABLED)
        self.question_button.config(state=tk.DISABLED)
        self.question_window.config(state=tk.DISABLED)
        self.proof_window.config(state=tk.NORMAL)
        self.add_proof_button.config(state=tk.NORMAL)
        self.status_bar.config(text=f"Status: Problem has been uploaded. Please start your proof step by step.")

    def add_step_proof(self):
        self.steps += 1
        informal = self.proof_window.get()
        if informal == "QED":
            formal = "then show ?thesis sledgehammer"
        else:
            formal = self.model.step_proof_formalization(self.informal_question, self.formal_question, self.proofs, informal)[-1]
            print(formal)
            if "sledgehammer" not in formal:
                formal = formal.split('by')[0].strip('.') + "sledgehammer"
        self.proofs.append({"informal": informal, "formal": formal})
        self.informal_proofs.append(informal)
        self.formal_proofs.append(formal)
        child_window = ChildWindow(self.scrollable_frame, self.steps, informal, formal, self)
        self.child_windows.append(child_window)
        child_window.pack(fill=tk.X)
        self.proof_window.delete(0, tk.END)

        self.update_content()
        self.lock()
        self.status_bar.config(text=f"Status: New proof step has been appended. Please check your proof.")

    def update_content(self):
        self.content_window.config(state=tk.NORMAL)
        self.content_window.delete('1.0', tk.END)
        self.content_window.insert(tk.END, "PROBLEM:\n")
        self.content_window.insert(tk.END, self.informal_question+'\n')
        self.content_window.insert(tk.END, "PROOFS:")
        for child_window in self.child_windows:
            self.content_window.insert(tk.END, '\n'+child_window.informal)
            line = len(self.content_window.get('1.0', tk.END).splitlines())
            self.content_window.highlight_line(line, child_window.status)

    def lock(self):
        for i in range(self.steps):
            if self.child_windows[i].step < self.steps:
                self.child_windows[i].lock()

    def unlock(self):
        for i in range(self.steps):
            if self.child_windows[i].step == self.steps:
                self.child_windows[i].unlock()

    def remove_child_window(self, child_window):
        self.steps -= 1
        self.child_windows.remove(child_window)
        child_window.destroy()
        self.informal_proofs.pop()
        content = f"PROBLEM:\n{self.informal_question}\n\nPROOFS:\n{self.join_proof(self.informal_proofs)}"
        self.content_window.config(state=tk.NORMAL)
        self.content_window.delete('1.0', tk.END)
        self.content_window.insert(tk.END, content)
        self.content_window.config(state=tk.DISABLED)
        self.unlock()

    @staticmethod
    def join_proof(informal_proofs):
        return "\n".join(informal_proofs)


if __name__ == "__main__":
    main_window = MainWindow()
    main_window.mainloop()

