import os
import sys
import time
import shutil
from playwright.sync_api import sync_playwright

DOCUMENTS_DIR = "v:/PROJECTS/potens-intern-ai-vaidik-pipaliya/documents"
BRAIN_DIR = r"C:\Users\vaidik\.gemini\antigravity-ide\brain\54441ec4-1825-460f-9d51-7e6fe20d24ca"

def setup_temp_docs():
    print("Setting up temporary test documents...")
    doc1_content = (
        "Potens Internship Program Policy document version 1.0.\n"
        "The weekly working commitment is exactly 40 hours per week.\n"
        "The program duration is set to 3 months for all interns.\n"
        "Stipend is paid monthly at a rate of 15,000 INR."
    )
    doc2_content = (
        "Potens Internship Program Policy document version 2.0 (Revised).\n"
        "The weekly working commitment is exactly 45 hours per week due to training overhead.\n"
        "The program duration is set to 3 months for all interns.\n"
        "Stipend is paid monthly at a rate of 18,000 INR."
    )
    
    doc1_path = os.path.join(DOCUMENTS_DIR, "temp_policy_v1.txt")
    doc2_path = os.path.join(DOCUMENTS_DIR, "temp_policy_v2.txt")
    
    with open(doc1_path, "w", encoding="utf-8") as f:
        f.write(doc1_content)
    with open(doc2_path, "w", encoding="utf-8") as f:
        f.write(doc2_content)
        
    print("Temporary documents created.")
    return doc1_path, doc2_path

def cleanup_temp_docs(doc1_path, doc2_path):
    print("Cleaning up temporary test documents...")
    if os.path.exists(doc1_path):
        os.remove(doc1_path)
    if os.path.exists(doc2_path):
        os.remove(doc2_path)
    print("Temporary documents cleaned.")

def run_ui_test():
    doc1_path, doc2_path = setup_temp_docs()
    
    try:
        with sync_playwright() as p:
            print("Launching headless Chromium browser...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 960})
            page = context.new_page()
            
            print("Navigating to Streamlit app at http://localhost:8501...")
            page.goto("http://localhost:8501", timeout=30000)
            page.wait_for_timeout(3000)
            
            # Take initial page screenshot
            page.screenshot(path=os.path.join(BRAIN_DIR, "ui_initial.png"))
            print("Initial page screenshot captured.")
            
            # Step 1: Rebuild Vector Store via UI
            print("Clicking 'Rebuild Vector Store' button...")
            rebuild_btn = page.get_by_role("button", name="🔄 Rebuild Vector Store")
            if rebuild_btn.is_visible():
                rebuild_btn.click()
                print("Rebuild button clicked. Waiting for indexing to complete...")
                # Wait for Success message to appear
                page.wait_for_selector("text=Successfully indexed", timeout=60000)
                print("Vector store rebuilt successfully through UI!")
            else:
                print("Rebuild button not found, skipping UI rebuild.")
                
            page.screenshot(path=os.path.join(BRAIN_DIR, "ui_after_rebuild.png"))
            
            # Step 2: Test Q&A Tab
            print("Testing Q&A Tab...")
            # Streamlit inputs are typically st.text_input, let's find the text input
            query_input = page.get_by_label("Enter your question:")
            query_input.fill("What is the stipend amount in v1?")
            
            print("Submitting question...")
            page.get_by_role("button", name="🚀 Answer Question").click()
            
            # Wait for answer to appear
            print("Waiting for answer...")
            page.wait_for_selector("text=15,000 INR", timeout=30000)
            print("Answer rendered correctly!")
            
            # Wait 2s for animations to finish
            page.wait_for_timeout(2000)
            
            # Capture Q&A response screenshot
            qa_screenshot_path = os.path.join(BRAIN_DIR, "ui_qa_response.png")
            page.screenshot(path=qa_screenshot_path)
            print(f"Q&A response screenshot saved to: {qa_screenshot_path}")
            
            # Step 3: Switch to Contradiction Auditor Tab
            print("Switching to Document Contradiction Auditor tab...")
            page.get_by_role("tab", name="⚖️ Document Contradiction Auditor").click()
            page.wait_for_timeout(2000)
            
            # Select Document 1 (selectbox)
            print("Selecting Document 1...")
            # We locate by label or find the selectbox
            # Streamlit lists selectbox inside a container, let's locate by label
            doc1_select = page.get_by_label("Document 1:")
            doc1_select.click()
            page.wait_for_timeout(500)
            # Click the option temp_policy_v1.txt in the dropdown list
            page.get_by_role("option", name="temp_policy_v1.txt").first.click()
            page.wait_for_timeout(1000)
            
            # Select Document 2 (selectbox)
            print("Selecting Document 2...")
            doc2_select = page.get_by_label("Document 2:")
            doc2_select.click()
            page.wait_for_timeout(500)
            page.get_by_role("option", name="temp_policy_v2.txt").first.click()
            page.wait_for_timeout(1000)
            
            # Enter topic
            print("Entering topic 'stipend'...")
            topic_input = page.get_by_label("Audited Topic (e.g. stipend, duration):")
            topic_input.fill("stipend")
            page.wait_for_timeout(500)
            
            # Run Version Audit
            print("Clicking 'Run Version Audit' button...")
            page.get_by_role("button", name="⚖️ Run Version Audit").click()
            
            print("Waiting for audit results...")
            page.wait_for_selector("text=CONTRADICTION DETECTED", timeout=30000)
            print("Contradiction successfully detected by the auditor!")
            
            # Wait 2s for rendering
            page.wait_for_timeout(2000)
            
            # Capture Contradiction screenshot
            contradict_screenshot_path = os.path.join(BRAIN_DIR, "ui_contradiction_response.png")
            page.screenshot(path=contradict_screenshot_path)
            print(f"Contradiction screenshot saved to: {contradict_screenshot_path}")
            
            browser.close()
            print("Browser test completed successfully.")
            
    finally:
        cleanup_temp_docs(doc1_path, doc2_path)
        
        # Restore database to original state
        print("Restoring database to clean state...")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()
                page.goto("http://localhost:8501")
                page.wait_for_timeout(3000)
                rebuild_btn = page.get_by_role("button", name="🔄 Rebuild Vector Store")
                if rebuild_btn.is_visible():
                    rebuild_btn.click()
                    page.wait_for_selector("text=Successfully indexed", timeout=60000)
                    print("Vector store restored to clean state.")
                browser.close()
        except Exception as e:
            print(f"Warning during final database cleanup: {e}")

if __name__ == "__main__":
    run_ui_test()
