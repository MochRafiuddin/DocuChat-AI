from flask import Blueprint, request, jsonify
from app.services import DocumentService, VectorStoreService, ChatService

ask_bp = Blueprint('ask', __name__)

# Inisialisasi services
document_service = DocumentService()
vector_store_service = VectorStoreService()
chat_service = ChatService()

@ask_bp.route('/api/ask', methods=['POST'])
def process_and_ask():
    # Cek apakah ada file yang diupload
    has_files = 'files' in request.files
    # Cek apakah ada pertanyaan
    data = request.get_json() if request.is_json else {}
    question = data.get('question') if data else None
    
    # Jika tidak ada file dan tidak ada pertanyaan
    if not has_files and not question:
        return jsonify({
            'error': 'Request harus menyertakan file untuk dianalisis atau pertanyaan untuk dijawab'
        }), 400
    
    # Proses file jika ada
    if has_files:
        files = request.files.getlist('files')
        all_texts = []
        
        for file in files:
            texts = document_service.process_file(file)
            if texts:
                all_texts.extend(texts)
        
        if not all_texts:
            return jsonify({
                'error': 'Tidak ada dokumen valid yang dapat diproses'
            }), 400
        
        vector_store_service.create_vector_store(all_texts)
        
        # Jika file berhasil diupload, langsung analisis isinya
        
        if not question:
            default_question = "Berikan ringkasan dari dokumen ini"
            retriever = vector_store_service.get_retriever()
            chain = chat_service.create_chain(retriever)
            summary = chat_service.get_response(chain, default_question)
            return jsonify({
                'message': 'File berhasil diproses',
                'summary': summary
            })
    
    # Proses pertanyaan spesifik jika ada
    if question:
        retriever = vector_store_service.get_retriever()
        chain = chat_service.create_chain(retriever) if retriever else None
        response = chat_service.get_response(chain, question)
        
        return jsonify({
            'answer': response
        })