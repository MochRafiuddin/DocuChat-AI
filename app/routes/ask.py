from flask import Blueprint, request, jsonify
from app.services import DocumentService, VectorStoreService, ChatService
from app.config import EMBED_MODE, OPENAI_API_KEY
import openai
import traceback
import time
from openai import OpenAI

ask_bp = Blueprint('ask', __name__)

# Inisialisasi services
document_service = DocumentService()
vector_store_service = VectorStoreService()
chat_service = ChatService()
client = OpenAI(api_key=OPENAI_API_KEY)

@ask_bp.route('/api/ask', methods=['POST'])
def process_and_ask():
    try:
        # Cek apakah ada file yang diupload
        has_files = 'files' in request.files
        
        # Cek pertanyaan dari form-data atau json
        question = None
        if request.is_json:
            data = request.get_json()
            question = data.get('question')
        else:
            question = request.form.get('question')
        
        # Jika tidak ada pertanyaan
        if not question:
            return jsonify({
                'error': 'Mohon berikan pertanyaan yang ingin dijawab'
            }), 400
        
        # Proses file jika ada
        if has_files:
            files = request.files.getlist('files')
            all_texts = []
            raw_text = ""
            
            for file in files:
                texts = document_service.process_file(file)
                if texts:
                    # Simpan raw text untuk mode tanpa embedding
                    raw_text += "\n\n".join([doc.page_content for doc in texts])
                    # Simpan texts untuk mode dengan embedding
                    all_texts.extend(texts)
            
            if not all_texts:
                return jsonify({
                    'error': 'Tidak ada teks yang bisa diekstrak dari file'
                }), 400
            
            try:    
                start_embed = time.time()
                # Gunakan EMBED_MODE dari environment
                if EMBED_MODE.lower() == 'true':
                    # Mode dengan embedding
                    print("Menggunakan mode embedding sesuai konfigurasi")
                    vector_store_service.create_vector_store(all_texts)
                    print("Berhasil membuat vector store dari dokumen")
                    retriever = vector_store_service.get_retriever()
                    
                    # Gunakan model yang sama untuk mode embedding
                    prompt = f"You are an assistant. Based on the following content:\n\n{raw_text}\n\nAnswer this question:\n{question}"
                    response = client.chat.completions.create(
                        model="gpt-4.1-nano-2025-04-14",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                    ).choices[0].message.content.strip()
                else:
                    # Mode tanpa embedding (direct text)
                    print("Menggunakan mode direct text sesuai konfigurasi")
                    prompt = f"You are an assistant. Based on the following content:\n\n{raw_text[:100000]}\n\nAnswer this question:\n{question}"
                    response = client.chat.completions.create(
                        model="gpt-4.1-nano-2025-04-14",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                    ).choices[0].message.content.strip()
                
                print(f"✅ pertanyaan selesai dalam {time.time() - start_embed:.2f} detik")
            except openai.BadRequestError as e:
                if "maximum context length" in str(e):
                    return jsonify({
                        'error': 'Dokumen terlalu panjang. Silakan:\n1. Gunakan mode embedding (atur EMBED_MODE=true di .env), atau\n2. Upload dokumen yang lebih pendek'
                    }), 400
                return jsonify({
                    'error': f'Error dari OpenAI: {str(e)}'
                }), 400
            except Exception as e:
                print(f"Error saat memproses: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                return jsonify({
                    'error': 'Gagal memproses permintaan'
                }), 400
        else:
            # Tidak ada file, langsung tanya ke LLM
            prompt = f"You are an assistant. Answer this question:\n{question}"
            response = client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            ).choices[0].message.content.strip()
        
        return jsonify({
            'answer': response
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Terjadi kesalahan internal server'
        }), 500