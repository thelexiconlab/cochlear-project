def switch_simdrop(fluency_list, semantic_similarity):
    '''
        Similarity Drop Switch Method from Hills TT, Jones MN, Todd (2012).
        
        Args:
            fluency_list (list, size = L): fluency list to predict switches on
            semantic_similarity (list, size = L): a list of semantic similarities between items in the fluency list
        Returns:
            a list, size L, of switches, where 0 = no switch, 1 = switch, 2 = boundary case
    '''
    simdrop = []
    for k in range(len(fluency_list)):
        if (k > 0 and k < (len(fluency_list)-1)): 
            # simdrop
            if (semantic_similarity[k+1] > semantic_similarity[k]) and (semantic_similarity[k-1] > semantic_similarity[k]):
                simdrop.append(1)
            
            else:
                simdrop.append(0)

        else:
            simdrop.append(2)

    return simdrop
