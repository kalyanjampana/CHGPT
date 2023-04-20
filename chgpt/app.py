import gradio as gr
import configparser
import requests
import json
import pickle
from base64 import b64encode
import fitz
import os
import pytesseract
import numpy as np
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback

config = configparser.ConfigParser()
config.read('config.ini')
chkey = config['chapi']['CH_API_KEY']
token = b64encode(f"{chkey}".encode('utf-8')).decode("ascii")
#os.environ["TESSDATA_PREFIX"] = "/Users/kalyan/tesseract/tessdata/"
#os.environ["OPENAI_API_KEY"] = config['openai']['OPENAI_API_KEY']

with gr.Blocks(gr.themes.Soft()) as demo:
    # Create input Text search box
    input_box = gr.Textbox(label="Input search string for a UK Company Name")
    # State variable to store the Document ID
    doc_id = gr.State()
    # Button to initiate the Company Search
    search_btn = gr.Button("Search")
    # Column to display the Company search result - List of companies
    with gr.Column(visible=False) as output_col:
        company_list_box = gr.Radio(choices=["Test1","Test2"],label="Company search result")
        #company_list = gr.DataFrame(headers=['Company Number','Company Name','Date Created','Registered Address'], interactive=False)
    
    # Text box to display the latest filing information
    display_filing = gr.Textbox(label="",interactive=False, visible=False)

    # Button to get the latest filing document
    submit_btn = gr.Button("Get latest filing", visible=False)

    # Text box Display the Document information
    display_filing_doc_info = gr.Textbox(label="",interactive=False, visible=False)

    # OpenAPI Key Input box
    openapi_key_input = gr.Textbox(label="OpenAI API Key",interactive=True, visible=False)

    # Button to initiate the processing of the Document - OPENAI Call initiated here too
    process_filing_btn = gr.Button("Summarize the Account filing", visible=False)

    # Text box showing the progress of the processing and the message
    processed_info = gr.Textbox(label="",interactive=False, visible=False)

    # Final Text box to display the summary of the Annual Accounts
    summary_text = gr.Textbox(label="Summary using OPENAI",interactive=False, visible=False)

    # Clear all fields to start again
    clear = gr.Button("Clear")

    # Function that does the Company search based on a search string. Gets the top 10 results 
    def company_search(text):
        url = "https://api.company-information.service.gov.uk/advanced-search/companies?company_name_includes=" + text + "&company_status=active&size=10"
        auth = f'Basic {token}'
        payload={}
        headers = {
            'Authorization': auth
        }
        response = requests.request("GET", url, headers=headers, data=payload)
        select_resp = []
        if response.status_code == 200:
            resp = json.loads(response.text)
            for comp in resp["items"]:
                addr = []
                for key, value in comp["registered_office_address"].items():
                    addr.append(value) 
                select_resp.append(comp["company_number"] + " : " + comp["company_name"] + " : " + ', '.join(addr))
            return {output_col: gr.update(visible=True), company_list_box: gr.update(choices=select_resp,interactive=True)}
        else:
            select_resp.append("No matching companies found")
            return {output_col: gr.update(visible=True), company_list_box: gr.update(choices=select_resp,interactive=False)}            
    
    # Function to get the Filing information of a selected company
    def company_selected(selected_company, docid):
        regid = selected_company.split(' : ')[0]
        filings_url = "https://api.company-information.service.gov.uk/company/" + regid + "/filing-history?category=accounts&items_per_page=1"
        print(filings_url)
        auth = f'Basic {token}'
        payload={}
        headers = {
            'Authorization': auth
        }
        response = requests.request("GET", filings_url, headers=headers, data=payload)
        resp = json.loads(response.text)
        #print(resp["items"])
        if len(resp["items"])>0:
            resp_value = f'Latest filing done on {resp["items"][0]["date"]}.'
            if "links" in resp["items"][0]:
                if "document_metadata" in resp["items"][0]["links"]:
                    docid = resp["items"][0]["links"]["document_metadata"].rsplit('/',1)[-1]
                    return {display_filing: gr.Textbox.update(visible=True, value=resp_value), submit_btn: gr.update(visible=True), doc_id : docid}
                else:
                    resp_value += "But Document Metadata is not available."
                    return {display_filing: gr.Textbox.update(visible=True, value=resp_value), submit_btn: gr.update(visible=False), doc_id : "None"}
            else:
                resp_value += "But Links to the filing not available."
                return {display_filing: gr.Textbox.update(visible=True, value=resp_value), submit_btn: gr.update(visible=False), doc_id : "None"}
        else:
            return {display_filing: gr.Textbox.update(visible=True, value="No record of accounts filed for the company"), submit_btn: gr.update(visible=False), doc_id : "None"}

    # Function to get the Filing document related to the latest Annual Account filing
    def get_filing(docid):
        doc_url = "https://document-api.company-information.service.gov.uk/document/" + docid + "/content"
        print(doc_url)
        auth = f'Basic {token}'
        payload={}
        headers = {
            'Authorization': auth
        }
        response = requests.request("GET", doc_url, headers=headers, data=payload)
        #print(response.text)
        content_type = response.headers['Content-Type']
        print(content_type)
        resp_value = f'Filing document is of type {content_type}. '
        if content_type == 'application/pdf':
            filename = f'doc_{docid}.pdf'
            filepath = './data/'+filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            pdf_document = fitz.open(filepath)
            resp_value += f'PDF saved as: {filename}. There are a total of {pdf_document.page_count} pages'
            print(resp_value)
            return {display_filing_doc_info: gr.Textbox.update(visible=True, value=resp_value), process_filing_btn: gr.update(visible=True), openapi_key_input: gr.update(visible=True), processed_info: gr.update(visible=True), doc_id : docid}
        elif content_type == 'application/xhtml+xml':
            print('Work in progress to process these type of filings')
            resp_value += 'Work in progress to process these type of filings'
            return {display_filing_doc_info: gr.Textbox.update(visible=True, value=resp_value), process_filing_btn: gr.update(visible=False), openapi_key_input: gr.update(visible=False), processed_info: gr.update(visible=False), doc_id : "None"}
        else:
            print('Work in progress to process these type of filings')
            resp_value += 'Work in progress to process these type of filings'
            return {display_filing_doc_info: gr.Textbox.update(visible=True, value=resp_value), process_filing_btn: gr.update(visible=False), openapi_key_input: gr.update(visible=False), processed_info: gr.update(visible=False), doc_id : "None"}

    # Function to initial the Langchain chain with call to OPENAI to Summarize the Annual report
    def langchain_summarize(contents,openai_api_key):
        concatenated_content = '`n`n'.join(contents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=20,length_function=len)
        docs = text_splitter.create_documents([concatenated_content])
        prompt_template = """You are a financial analyst, analyzing the Annual report submitted by the limited company at UK Companies House. Write a concise summary of the following report:


        {text}

        
        CONCISE SUMMARY:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        #docs = [Document(page_content=t) for t in contents]
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=PROMPT)
        with get_openai_callback() as cb:
            resp = chain.run(docs)
            tkn_text = f'*** Spent a total of {cb.total_tokens} tokens ***'
        return resp, tkn_text

    # Function to extract text from the document and call the LangChain processing function with the text array of pages
    def process_filing(docid, openai_api_key, progress=gr.Progress()):
        progress(0,desc="Starting...")
        filepath = f'./data/doc_{docid}.pdf'
        pdf_document = fitz.open(filepath)
        text_path = f'./text/doc_{docid}.pkl'
        if os.path.exists(text_path):
            # retrieve list from file
            with open(text_path, 'rb') as f:
                contents = pickle.load(f)
        else:
            zoom_x = 2.0  # horizontal zoom
            zoom_y = 2.0  # vertical zoom
            mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
            contents = []
            for page in progress.tqdm(pdf_document, desc="Processing pages from PDF..."):
                # Convert the page to a PNG image using PyMuPDF
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height,pix.width, pix.n)
                texts = pytesseract.image_to_string(img)
                contents.append(texts)
            # save list to file
            with open(text_path, 'wb') as f:
                pickle.dump(contents, f)
        #print(contents)
        resp_value = f'Total of {pdf_document.page_count} pages processed. '
        summary_path = f'./summary/doc_{docid}.txt'
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = f.read()
            print(resp_value)
            return {processed_info: gr.Textbox.update(visible=True, value=resp_value), summary_text: gr.Textbox.update(visible=True, value=summary)}
        else:
            #progress(track_tqdm=True,desc="Calling OpenAI to summarize...")
            try:
                summary, tkn_text = langchain_summarize(contents, openai_api_key)
                resp_value += tkn_text
                with open(summary_path, 'wb') as f:
                    f.write(summary.encode())
                return {processed_info: gr.Textbox.update(visible=True, value=resp_value), summary_text: gr.Textbox.update(visible=True, value=summary)}
            except Exception as e:
                resp_value += 'LLM Call failed. Please check the OpenAI key again'
                return {processed_info: gr.Textbox.update(visible=True, value=resp_value), summary_text: gr.Textbox.update(visible=False)}
            finally:
                print(resp_value)
    
    def clear_screen():
        return {output_col: gr.update(visible=False),display_filing: gr.Textbox.update(visible=False),submit_btn: gr.update(visible=False), display_filing_doc_info:gr.update(visible=False), process_filing_btn:gr.update(visible=False),openapi_key_input:gr.update(visible=False),processed_info:gr.update(visible=False),summary_text:gr.update(visible=False)}

    search_btn.click(company_search,input_box,[company_list_box,output_col])
    company_list_box.change(company_selected,[company_list_box, doc_id],[display_filing, submit_btn, doc_id])
    submit_btn.click(get_filing,doc_id,[display_filing_doc_info, process_filing_btn, openapi_key_input, processed_info, doc_id])
    process_filing_btn.click(process_filing,[doc_id,openapi_key_input],[processed_info,summary_text])
    clear.click(clear_screen, None, [output_col,display_filing,submit_btn,display_filing_doc_info,process_filing_btn,openapi_key_input,processed_info,summary_text])

demo.queue(concurrency_count=2)
demo.launch()   