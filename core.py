from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from openai import OpenAI
from isabelle_client import start_isabelle_server, get_isabelle_client
import logging
import os
import json
import re
import time
from func_timeout import func_set_timeout
from huggingface_hub import login

ACCESS_TOKEN = None
access_token = ACCESS_TOKEN

# Check if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current running on {device}")
print(f"{torch.cuda.device_count()} GPU is running")

# LLM definition
class LLM:
    def __init__(self, name="llama3 8B"):
        self.name = name
        self.model_load(name)

    def model_load(self, name):
        # Load llama3 8B
        if name == "llama3 8B":
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', token=access_token)
            model = AutoModelForCausalLM.from_pretrained(
                'meta-llama/Meta-Llama-3-8B-Instruct',
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.tokenizer = tokenizer
            self.model = model

        # Load Qwen2 7B
        elif name == "Qwen2 7B":
            tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B-Instruct', token=access_token)
            model = AutoModelForCausalLM.from_pretrained(
                'Qwen/Qwen2-7B-Instruct',
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.tokenizer = tokenizer
            self.model = model

        # Load GLM4 9B
        elif name == "GLM4 9B":
            tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                "THUDM/glm-4-9b-chat",
                # torch_dtype=torch.bfloat16,
                load_in_4bit=True,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()
            self.tokenizer = tokenizer
            self.model = model
        else:
            raise Exception('Failed to load LLM')

    def problem_formalization(self, informal_problem):
        # This part is responsible for the formalization of natural language problem

        problem_prompt = f"""Please translate the following informal problem into corresponding isabelle language formal problem. Just translate do not provide any other text.
            Example:
            # Informal Problem:
            Suppose $ m, n, k, t $ are real number such as $m=2*n-n^2$ and $k = t^2+2*t+4$, prove that $k - m >= 2$.

            # Formal Problem:
            theorem algebra_test1:
              fixes m n k t :: real
              assumes h0: "m = 2*n - n^2"
                and h1: "k = t^2 + 2*t + 4"
              shows "k - m \<ge> 2"
            """
        if self.name == "llama3 8B":
            problem_input = [{"role": "system", "content": problem_prompt},
                             {"role": "informal", "content": informal_problem}]
            input_ids = self.tokenizer.apply_chat_template(
                problem_input,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            formal_problem = self.tokenizer.decode(response, skip_special_tokens=True)
            formal_problem = formal_problem.split("# Formal:\n")[-1]
            return formal_problem

        elif self.name == "Qwen2 7B":
            problem_input = [{"role": "system", "content": problem_prompt},
                             {"role": "informal", "content": informal_problem}]
            text = self.tokenizer.apply_chat_template(
                problem_input,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            formal_problem = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(formal_problem)
            formal_problem = formal_problem.split("# Formal:\n")[-1]
            formal_problem = formal_problem.split("```isar")[-1]
            formal_problem = formal_problem.split("```isabelle")[-1]
            formal_problem = formal_problem.split("```")[0]
            print(formal_problem)
            formal_problem = formal_problem.split("proof -")[0]
            print(formal_problem)
            return formal_problem

        elif self.name == "GLM4 9B":
            problem_input = [{"role": "system", "content": problem_prompt},
                             {"role": "informal", "content": informal_problem}]
            inputs = self.tokenizer.apply_chat_template(problem_input,
                                                   add_generation_prompt=True,
                                                   tokenize=True,
                                                   return_tensors="pt",
                                                   return_dict=True
                                                   )
            inputs = inputs.to(device)
            gen_kwargs = {"temperature": 0.9, "max_new_tokens": 512, "do_sample": True, "top_k": 2}
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            formal_problem = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            formal_problem = formal_problem.split("# Formal:\n")[-1]
            formal_problem = formal_problem.split("# Formal Problem:\n")[-1]
            formal_problem = formal_problem.split("<|informal|>")[0]
            formal_problem = formal_problem.split("proof")[0]
            formal_problem = formal_problem.replace('```', '')
            formal_problem = formal_problem.replace('isabelle', '')
            formal_problem = formal_problem.replace('定理', 'theorem')
            print(formal_problem)
            return formal_problem

    def full_proof_formalization(self, informal_problem, formal_problem, informal_proof):
        # Full-Proof formalization strategy

        proof_prompt = f"""Please translate the following informal proof into corresponding isabelle language formal proof. Just translate do not provide any other text. Prove all steps with sledgehammer.

            ### Informal Problem:
            Let $a, b, n$  be integers. Prove that if $a | n$ and $b | n$ with $gcd(a, b) = 1$ then $ab | n$.
            
            ### Informal Proof:
            Since $a|n$, $n$ could be rewritten as $(n/a)*a$. Therefore, $b|n$ is equal to $b|(n/a)*a$. Since $gcd(a,b)=1$ and $b|(n/a)*a$, it means that $b|(n/a)$. By multiple $a$ on the both side of $b|(n/a)$, we will know that $b*a|(n/a)*a$ and we can get $a*b|n$.
            
            ### Formal Problem:
            theorem number_theory_test1:
              fixes a b n :: int 
              assumes h0 : "a \<noteq> 0"
                and h1: "b \<noteq> 0"
                and h2: "a dvd n"
                and h3: "b dvd n"
                and h4: "gcd a b = 1"
              shows "a * b dvd n"
            
            ### Formal Proof:
            proof -
              (*Since $a|n$, $n$ could be rewritten as $(n/a)*a$.*)
              have h5: "n = (n div a) * a" using h0 h2 sledgehammer
              (*Therefore, $b|n$ is equal to $b|(n/a)*a$.*)
              have h6: "b dvd (n div a) * a" using h3 h5 sledgehammer
              (*Since $gcd(a,b)=1$ and $b|(n/a)*a$, it means that $b|(n/a)$.*)
              have h7: "b dvd (n div a)" using h4 h6 sledgehammer
              (*By multiple $a$ on the both side of $b|(n/a)$, we will know that $b*a|(n/a)*a$ and we can get $a*b|n$.*)
              have h8: "b * a dvd (n div a) * a" using h7 sledgehammer
              then show ?thesis  using h0 h7 h8 sledgehammer
            qed
        
            Now translate based on following problem:
    
            # Informal Problem:
            {informal_problem}
        
            # Formal Problem:
            {formal_problem}
        """
        if self.name == "llama3 8B":
            messages = [{"role": "system", "content": proof_prompt}, {"role": "Informal Proof", "content": informal_proof}]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response, skip_special_tokens=True)
            formal_proof = response.split("### Formal Proof:")[-1].strip()
            return response, formal_proof

    def step_proof_formalization(self, informal_problem, formal_problem, proofs, informal_proof):
        # Step-Proof formalization strategy

        proof_prompt = f"""Please translate the following informal proof step into corresponding isabelle language formal proof step. Just translate do not provide any other text.

        # Example:
        informal: Obviously, $m=2*n-n^2$ could be rewritten as $m=1-1+2*n-n^2$
        formal: have h2: "m=1-1+2*n-n^2" using h0 by auto

        Now translate based on following problem:

        # Informal Problem:
        {informal_problem}

        # Formal Problem:
        {formal_problem}
        """
        if self.name == "llama3 8B":
            messages = [
                {"role": "system", "content": proof_prompt},
            ]
            for proof in proofs:
                messages.append({"role": "informal", "content": proof['informal']})
                messages.append({"role": "formal", "content": proof['formal']})
            messages.append({"role": "informal", "content": informal_proof})

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = self.model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(response, skip_special_tokens=True)
            formal_step = response.split("# Formal:\n")[-1]
            formal_step = formal_step.split("formal:")[-1]
            formal_step = formal_step.split('show ?thesis')[0].strip()
            return response, formal_step

        elif self.name == "Qwen2 7B":
            messages = [
                {"role": "system", "content": proof_prompt},
            ]
            for proof in proofs:
                messages.append({"role": "informal", "content": proof['informal']})
                messages.append({"role": "formal", "content": proof['formal']})
            messages.append({"role": "informal", "content": informal_proof})

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(response)
            formal_step = response.split("# Formal:\n")[-1]
            formal_step = formal_step.split("formal:")[-1]
            formal_step = formal_step.split('show ?thesis')[0].strip()
            formal_step = formal_step.split("```isar")[-1]
            formal_step = formal_step.split("```isabelle")[-1]
            formal_step = formal_step.split("```")[0]
            formal_step = formal_step.split("formal:\n")[-1]
            formal_step = formal_step.split("formal\n")[-1]
            print(formal_step)
            formal_step = formal_step.split("qed")[0]
            return response, formal_step

        elif self.name == "GLM4 9B":
            messages = [
                {"role": "system", "content": proof_prompt},
            ]
            for proof in proofs:
                messages.append({"role": "informal", "content": proof['informal']})
                messages.append({"role": "formal", "content": proof['formal']})
            messages.append({"role": "informal", "content": informal_proof})
            inputs = self.tokenizer.apply_chat_template(messages,
                                                   add_generation_prompt=True,
                                                   tokenize=True,
                                                   return_tensors="pt",
                                                   return_dict=True
                                                   )
            inputs = inputs.to(device)
            gen_kwargs = {"temperature": 0.9, "max_new_tokens": 256, "do_sample": True, "top_k": 2}
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            formal_step = response.split('formal:')[-1]
            formal_step = formal_step.replace('```', '')
            formal_step = formal_step.replace('isabelle', '')
            formal_step = formal_step.replace('<|informal|>', '')
            print(formal_step)

            return response, formal_step

class Checker:
    def __init__(self):
        self.isabelle = self.connect()

    def connect(self):
        # Set up isabelle client
        try:
            server_info, _ = start_isabelle_server(
                name="test", port=9999, log_file="server.log"
            )
            isabelle = get_isabelle_client(server_info)
            isabelle.logger = logging.getLogger()
            isabelle.logger.setLevel(logging.INFO)
            isabelle.logger.addHandler(logging.FileHandler("session.log"))
            return isabelle
        except:
            raise Exception('Connection failed.')

    def get_line(self, pos):
        # Find the formal content based on line
        with open(f'TEMP.thy', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        content = lines[pos - 1].strip()
        return content

    def upload(self, formal):
        # Upload formal content into checker
        with open(f'TEMP.thy', 'w', encoding='utf-8') as f:
            body = formal
            head = "theory TEMP\nimports Complex_Main\nbegin\n"
            tail = "\nend"
            content = head + body + tail
            f.writelines(content)

    def debug(self, response):
        # Feedback system
        response = response[-1].response_body
        response = json.loads(response)
        nodes = response['nodes'][0]
        messages = nodes['messages']
        print(messages)
        if response['ok']:
            print('Proof succeed!')
            return 0
        for i in messages:
            if 'Failed to finish proof' in i['message']:
                pos = i['pos']['line']
                if self.get_line(pos) == 'qed':
                    print('Proof succeed at current step, please continue proof')
                    return 1
                else:
                    print('Proof Failed at current step')
                    return -1
        return -2

    def apply_sledgehammer(self, method, pos):
        # Apply sledgehammer found tactics into proper position
        with open(f'TEMP.thy', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines[pos - 1] = lines[pos - 1].replace('sledgehammer', f'{method}')
        with open(f'TEMP.thy', 'w', encoding='utf-8') as f:
            f.writelines(lines)

    # @func_set_timeout(900)
    def check(self):
        # Formal content verification

        response = self.isabelle.use_theories(theories=['TEMP'], master_dir=".")
        info = self.debug(response)
        response = response[-1].response_body
        response = json.loads(response)
        nodes = response['nodes'][0]
        messages = nodes['messages']
        method = None
        pos = None
        for i in messages:
            if 'Try this:' in i['message']:
                method = i['message'].split('Try this:')[-1]
                method = re.sub(r'\s*\(\d+(\.\d+)?\s*ms\)', '', method)
                pos = i['pos']['line']
                break
        if method and pos:
            self.apply_sledgehammer(method, pos)
            info = self.check()
            return info
        return info


class Pipeline:
    # Pipeline is used for testing

    def __init__(self, model_name='llama3 8B'):
        self.llm = LLM(name=model_name)
        self.isabelle = Checker()

    def full_proof(self, informal_problem, informal_proofs):
        formal_ts = []

        formal_st = time.time()
        formal_problem = self.llm.problem_formalization(informal_problem)
        formal_et = time.time()
        formal_t = formal_et - formal_st
        formal_ts.append(formal_t)

        informal_proof = ' '.join(informal_proofs)

        formal_st = time.time()
        response, formal_proof = self.llm.full_proof_formalization(informal_problem, formal_problem, informal_proof)
        formal_et = time.time()
        formal_t = formal_et - formal_st
        formal_ts.append(formal_t)

        formal = f'{formal_problem}\n{formal_proof}'
        self.isabelle.upload(formal)
        proof_st = time.time()
        info = self.isabelle.check()
        if info == 1:
            print('Failed to Finish Proof')
            step_r = 0
        elif info == 0:
            print('Proof Succeed')
            step_r = 1
        else:
            print(f'Proof Failed')
            step_r = 0
        proof_et = time.time()

        proof_t = proof_et - proof_st
        print(formal)

        formal_t = sum(formal_ts)

        return {'result': step_r, 'formal_time': formal_t, 'proof_time': proof_t,
                'response': response, 'informal_problem': informal_problem, 'formal_problem': formal_problem,
                'informal_proof': informal_proofs, 'formal_proof': formal_proof, 'formal_ts': formal_ts,
                'proof_ts': [proof_t]}


    def step_proof(self, informal_problem, informal_proofs):

        formal_ts = []
        proof_ts = []

        formal_st = time.time()
        formal_problem = self.llm.problem_formalization(informal_problem)
        formal_et = time.time()
        formal_ts.append(formal_et - formal_st)

        responses = []
        proofs = []
        N = len(informal_proofs) - 1
        step = 0
        for informal_proof in informal_proofs:
            step += 1
            if informal_proof == 'QED':
                formal_proof = 'then show ?thesis by auto'
            else:
                formal_st = time.time()
                response, formal_proof = self.llm.step_proof_formalization(informal_problem, formal_problem, proofs, informal_proof)
                formal_et = time.time()
                formal_ts.append(formal_et - formal_st)

                responses.append(response)
                if "sledgehammer" not in formal_proof:
                    formal_proof = formal_proof.split('by')[0].strip('.') + "sledgehammer"
            proofs.append({'informal': informal_proof, 'formal': formal_proof})
            formal = self.formal_parse(formal_problem, proofs)
            self.isabelle.upload(formal)

            proof_st = time.time()
            info = self.isabelle.check()
            proof_et = time.time()
            proof_ts.append(proof_et - proof_st)

            if info == 1:
                # proofs[step-1]['formal'] = formal_proof.replace('sledgehammer', 'sorry')
                pass
            elif info == 0:
                print('Proof Succeed')
                print(self.formal_parse(formal_problem, proofs))
                break
            else:
                print(f'Proof Failed at step{step}:{informal_proof}')
                print(self.formal_parse(formal_problem, proofs))
                break
        step_r = (step-1)/N
        formal_proof = self.formal_parse(formal_problem, proofs)

        formal_t = sum(formal_ts)
        proof_t = sum(proof_ts)
        response = ''.join(responses)

        return {'result': step_r, 'formal_time': formal_t, 'proof_time': proof_t,
                'response': response, 'informal_problem': informal_problem, 'formal_problem': formal_problem,
                'informal_proof': informal_proofs, 'formal_proof': formal_proof, 'formal_ts': formal_ts,
                'proof_ts': proof_ts}

    def formal_parse(self, problem, proofs):
        formal = f"{problem}\nproof-\n"
        # for proof in proofs:
        #     formal += f"(*{proof['informal']}*)\n{proof['formal']}\n"
        steps = len(proofs)
        for step in range(steps):
            if step != steps - 1:
                formal += f"(*{proofs[step]['informal']}*)\n{proofs[step]['formal'].replace('sledgehammer', 'sorry')}\n"
                step += 1
            else:
                formal += f"(*{proofs[step]['informal']}*)\n{proofs[step]['formal']}\n"
        formal += 'qed\n'
        return formal
