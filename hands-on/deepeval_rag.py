import os
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    GEval
)
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate

model = GeminiModel(
    "gemini-2.5-flash",
    api_key=os.getenv("API_KEY_LLM")
)

test_case = LLMTestCase(
    input="Apa penyebab utama kemacetan di Jakarta?",
    actual_output="Kemacetan di Jakarta disebabkan oleh semakin banyaknya kendaraan pribadi.",
    expected_output="Faktor utama kemacetan di Jakarta adalah pertumbuhan kendaraan pribadi yang tinggi dan rendahnya penggunaan transportasi umum.",
    retrieval_context=[
        "Studi pada tahun 2020 menunjukkan pertumbuhan kendaraan pribadi meningkat 8% per tahun di Jakarta.",# relevan
        "Transportasi umum masih belum menjadi pilihan mayoritas warga karena keterbatasan jangkauan.",# relevan
        "Sejarah Jakarta menunjukkan bahwa pada era kolonial, jalur kereta dan trem pernah digunakan.",# tidak relevan
        "Jakarta memiliki lebih dari 13 juta perjalanan harian dalam kota.",# sedikit relevan, tapi gak langsung
        "Kemacetan juga dipengaruhi oleh pembangunan infrastruktur yang memakan satu lajur jalan." #agak relevan
    ]
)

# apakah informasi hasil retrieval dipakai untuk menentukan jawaban
faithfulness = FaithfulnessMetric(
    model=model,
)

# apakah jawabannya "nyambung" dengan pertanyaan
relevance = AnswerRelevancyMetric(
    model=model,
)

# Perbandingan jawaban langsung
correctness = GEval(
    name="Correctness",
    model=model,
    criteria= "Tentukan apakah jawaban tersebut secara faktual benar dibandingkan dengan jawaban referensi.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
)

# metrik retrieval
# K tidak secara eksplisit ditentukan, tapi langsung dari
# panjang konteks di atas
# artinya K = 5
precision = ContextualPrecisionMetric(
    model=model,
)

recall = ContextualRecallMetric(
    model=model,
)


evaluate(
    test_cases=[test_case],
    metrics=[faithfulness, relevance, correctness, precision, recall],
)