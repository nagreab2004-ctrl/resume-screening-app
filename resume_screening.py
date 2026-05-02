from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample Resumes
resume1 = "Python developer with machine learning and data analysis skills"
resume2 = "Java developer with spring boot and backend development experience"
resume3 = "Data analyst skilled in SQL Python and visualization"
resume4 = "Web developer with HTML CSS JavaScript React"
resume5 = "Machine learning engineer with deep learning and AI"

# Job Description
job_desc = "Looking for Python developer with machine learning and data analysis"

documents = [resume1, resume2, resume3, resume4, resume5, job_desc]

# TF-IDF
tfidf = TfidfVectorizer()
matrix = tfidf.fit_transform(documents)

# Similarity
similarity = cosine_similarity(matrix[-1], matrix[:-1])

print("\nResume Screening Results:\n")

scores = similarity[0]

for i, score in enumerate(scores):
    print(f"Resume {i+1} Match Score: {round(score*100, 2)}%")

# Ranking
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

print("\nTop Candidates Ranking:\n")

for rank, (index, score) in enumerate(ranked):
    print(f"Rank {rank+1}: Resume {index+1} ({round(score*100, 2)}%)")