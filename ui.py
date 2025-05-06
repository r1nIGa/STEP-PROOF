# UI相关库
import tkinter as tk
from tkinter import filedialog

import core
# StepProof内核
from core import *

# PDF导出库
from fpdf import FPDF

import argparse


class PDF(FPDF):
    # 用于导出证明的PDF格式文件
    def chapter_header(self, header):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, header, 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, lines):
        self.set_font('Arial', '', 11)
        for line in lines:
            # 处理高亮背景色
            if line['bg_color']:
                self.set_fill_color(*line['bg_color'])
                self.multi_cell(0, 5, line['text'], fill=True)
            else:
                self.multi_cell(0, 5, line['text'])
            self.ln(2)


class HighlightText(tk.Text):
    # 将证明步骤根据状态进行高光
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

    def highlight_line_delete(self, line_number):
        tag = f'line{line_number}'
        self.tag_remove(tag, f'{line_number}.0', f'{line_number}.end')


class ChildWindow(tk.Frame):
    # 步骤证明子窗口
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

        self.edit_button = tk.Button(self.button_frame, text="EDIT", command=self.edit_formal)
        self.edit_button.pack(side=tk.LEFT)
        self.edit_button.pack_forget()  # 初始隐藏EDIT按钮

    def check_formal(self):
        if self.action_button2_hide:
            self.label.config(text=f'{self.informal}\n{self.formal}')
            self.action_button2_hide = False
            self.check_button.config(text="HIDE")
            self.edit_button.pack(side=tk.LEFT)  # 显示时添加EDIT按钮
        else:
            self.label.config(text=self.informal)
            self.action_button2_hide = True
            self.check_button.config(text="CHECK")
            self.edit_button.pack_forget()  # 隐藏时移除EDIT按钮

    def edit_formal(self):
        # 创建编辑对话框
        edit_dialog = tk.Toplevel(self)
        edit_dialog.title("Edit Formal Proof")

        # 添加文本输入框
        edit_text = tk.Text(edit_dialog, height=10, width=50)
        edit_text.pack(padx=10, pady=10)
        edit_text.insert(tk.END, self.formal)

        # 确认按钮
        def save_changes():
            new_formal = edit_text.get("1.0", tk.END).strip()
            self.formal = new_formal
            if not self.action_button2_hide:  # 如果当前显示形式化内容则更新显示
                self.label.config(text=f'{self.informal}\n{self.formal}')
            edit_dialog.destroy()

        save_button = tk.Button(edit_dialog, text="Save", command=save_changes)
        save_button.pack(pady=5)

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



class MainWindow(tk.Tk):
    # 主窗口设计

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

        self.model_name = "llama3 8B"

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
        # file_menu.add_command(label="Connect", command=self.connect)
        file_menu.add_command(label="Export PDF", command=self.export_to_pdf)  # 新增导出命令
        menu_bar.add_cascade(label="File", menu=file_menu)
        menu_bar.add_command(label="Connect", command=self.connect)

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
            child_window.delete_window()
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
        """更新内容时自动清除旧的高亮标签"""
        self.content_window.config(state=tk.NORMAL)
        self.content_window.delete('1.0', tk.END)

        # 清除所有现有高亮标签
        for tag in self.content_window.tag_names():
            self.content_window.tag_delete(tag)

        # 重新插入内容并应用高亮
        self.content_window.insert(tk.END, "PROBLEM:\n")
        self.content_window.insert(tk.END, self.informal_question + '\n')
        self.content_window.insert(tk.END, "PROOFS:")
        for child_window in self.child_windows:
            self.content_window.insert(tk.END, '\n' + child_window.informal)
            line = len(self.content_window.get('1.0', tk.END).splitlines())
            self.content_window.highlight_line(line, child_window.status)

        self.content_window.config(state=tk.DISABLED)

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
        self.child_windows.pop()  # 假设按顺序撤销，删除最后一个子窗口
        child_window.destroy()
        self.informal_proofs.pop()
        self.update_content()  # 调用update_content来更新内容和高亮
        self.unlock()

    @staticmethod
    def join_proof(informal_proofs):
        return "\n".join(informal_proofs)

    def export_to_pdf(self):
        # 获取所有带格式的内容
        proof_lines = self._get_highlighted_content()
        # 弹出保存对话框
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")]
        )
        if not file_path:
            return
        # 创建PDF文档
        pdf = PDF()
        pdf.add_page()
        # 添加问题部分
        pdf.chapter_title("PROBLEM")
        problem_lines = [{
            'text': self.informal_question,
            'bg_color': None
        }]
        pdf.chapter_body(problem_lines)
        # 添加证明步骤
        pdf.chapter_title("PROOFS")
        pdf.chapter_body(proof_lines)

        # 保存文件
        pdf.output(file_path)
        self.status_bar.config(text=f"Status: Exported to {file_path}")

    def _get_line_bg_color(self, line_num):
        """获取指定行的背景颜色（直接返回RGB元组）"""
        tags = self.content_window.tag_names(f"{line_num}.0")
        for tag in tags:
            if tag.startswith('line'):
                color = self.content_window.tag_cget(tag, "background")
                # 直接转换为RGB值
                rgb = self.content_window.winfo_rgb(color)
                # 将颜色值从0-65535范围转换为0-255范围
                return (rgb[0] // 256, rgb[1] // 256, rgb[2] // 256)
        return None

    def _get_highlighted_content(self):
        """从文本组件获取带高亮格式的内容"""
        self.content_window.config(state=tk.NORMAL)
        content = []

        total_lines = int(self.content_window.index('end-1c').split('.')[0])

        for line_num in range(1, total_lines + 1):
            line_text = self.content_window.get(f"{line_num}.0", f"{line_num}.end")
            bg_color = self._get_line_bg_color(line_num)  # 现在直接获取RGB元组

            content.append({
                'text': line_text.strip('\n'),
                'bg_color': bg_color
            })

        self.content_window.config(state=tk.DISABLED)
        idx = content.index({'text': 'PROOFS:', 'bg_color': None})
        content = content[idx + 1:]

        return content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Step proof is a natural language interactive theorem prover.')
    parser.add_argument('--isabelle_path', type=str, required=True, help='Input your isabelle bin path.')
    parser.add_argument('--access_token', type=str, required=True, help='Input your hugging-face access token.')

    args = parser.parse_args()
    core.ACCESS_TOKEN = args.access_token
    login(token=args.access_token)
    os.environ['PATH'] += f":{args.isabelle_path}"

    main_window = MainWindow()
    main_window.mainloop()
