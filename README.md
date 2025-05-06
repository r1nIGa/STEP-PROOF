# StepProof

Step proof is a natural language interactive theorem prover. It includes a complete interactive interface and can implement sentence-level formal verification of mathematical proofs in natural language.

## Installation

Install required python package.

``pip install -r requirements.txt``

Then, install isabelle from following website [Isabelle Official Website](https://isabelle.in.tum.de/index.html)

or using following command in linux

``wget https://isabelle.in.tum.de/dist/Isabelle2025_linux.tar.gz``

``tar -xzf Isabelle2025_linux.tar.gz``

Also make sure the cuda has been installed, since it's necessary for LLM loading. The cuda install guidance can be found in https://developer.nvidia.com/cuda-toolkit

Once cuda has been installed, you can use following command in terminal to check if it's available now.

``python -c "import torch; print(torch.cuda.is_available())"``

Then, to download and access the model, accessing permissions are needed. User can access following pages to get the authority of llama3 https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

Once get the authority,  you can find your hugging-face access token with following web page [User access tokens](https://huggingface.co/docs/hub/security-tokens)

Finally, you can download LLM in your PC with following command.

``python download.py --token <YOUR_HF_TOKEN> --model meta-llama/Meta-Llama-3-8B``

At this point, all the preparations required before running StepProof are ready.

## Run

The main program for starting StepProof is as follows:

``python ui.py --isabelle_path <YOUR_ISABELLE_PATH> --access_token <YOUR_HF_TOKEN>``

Once successfully started, you will see the following UI interface

![img.png](/img/img.png)


## Tutorial

After initialize the StepProof, user need to connect the StepProof with LLM via connect.

![connect.png](/img/connect.png)

When status shows has connected to llm, user can upload problem in upper input box and click **Upload Problem** to formalize the problem.

![upload_problem.png](/img/upload_problem.png)

Then the proof input box and button will unlock. User can submit their proof here step by step.

![upload_proof.png](/img/upload_proof.png)

Once the proof step has been formalized, it will show a child window to manage the proof and user can click **Proof** button to verify the proof.

![proof_management.png](/img/proof_management.png)

If the step has been verified in isabelle, the proof step in upper content box will be marked with green background to show it has been verified and user could input the next step.

![next_step.png](/img/next_step.png)

After all step has been verified, user could input **QED** to show the proof has been completed and system will verify if all the proof step could solve the global goal.

![finish_proof.png](/img/finish_proof.png)

Finally, we can export the verified proof into pdf file from the **File** in upper menu.

![pdf.png](/img/pdf.png)