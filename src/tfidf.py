from pyspark.sql import SparkSession
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

spark = SparkSession.builder.getOrCreate()

breitbart = spark.read.csv("scrapbriet.csv")

breitbart.describe()

bDocs = breitbart.select('_c0')

bContent = breitbart.select('_c2').rdd.map(lambda x:x['_c2']).filter(lambda y:y is not None).collect()

tfidf = TfidfVectorizer(max_df=0.5,min_df=2,stop_words='english')

X = tfidf.fit_transform(bContent)


km = KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(X)

for i in range(len(km.labels_)):
    print("document "+str(i)+" clusters to center : "+str(km.labels_[i]))