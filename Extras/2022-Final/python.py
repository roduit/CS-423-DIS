def knn_weighting_estimate(doc_vectors, doc_labels, query_vector, k=10):
    """ Weighting estimation for kNN classification
    :param doc_vectors: Document vectors (np.array(np.array))
    :param doc_labels: Document labels/topics (list)
    :param query_vector: Query vector (np.array)
    :param k: Number of nearest neighbors to retrieve
    
    :return: A dictionary containing the estimation score for each label/topic (dict)
    """
    # --------------
    # YOUR CODE HERE
    query_vector = np.array(query_vector)
    scores = {}
    
    top_k_docs = knn(doc_vectors, query_vector)
    distances = np.array([cosine_similarity(query_vector, doc_vectors[doc]) for doc in top_k_docs])

    for i in range(min(doc_labels),max(doc_labels)+1):
        retrieved_labels = np.array([int(doc_labels[idx] == i) for idx in top_k_docs])

        scores[i] = np.dot(distances, retrieved_labels)
    # --------------
    return scores