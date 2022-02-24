import numpy as np
cat_data = np.load("eden.npz")
drug_embedding = cat_data['drug_embed']
entity_embedding = cat_data['entity_embed']
relation_embedding = cat_data['relation_embed']

def get_score(h,r,t):
    return np.linalg.norm(h+r-t)

def get_schema():
    schema={}
    schema["drug"]={
        "neighbor":["dis"],
        "st":0,
        "ed":0
    }
    schema["dis"]={
        "neighbor":["pheno","gene","drug"],
        "st":0,
        "ed":17093
    }
    schema["pheno"]={
        "neighbor":["dis"],
        "st":180118,
        "ed":186595
    }
    schema["gene"]={
        "neighbor":["dis","go","path","pro"],
        "st":39190,
        "ed":153661
    }
    schema["path"]={
        "neighbor":["gene"],
        "st":153662,
        "ed":156024
    }
    schema["go"]={
        "neighbor":["gene"],
        "st":156025,
        "ed":180117
    }
    schema["pro"]={
        "neighbor":["gene","pro"],
        "st":17094,
        "ed":39189
    }
    return schema

def get_link():
    link={}
    link["drugdis"]={
        "st":0,
        "ed":0
    }
    link["disdrug"]={
        "st":0,
        "ed":0
    }
    link["disgene"]={
        "st":1,
        "ed":1
    }
    link["genedis"]={
        "st":1,
        "ed":1
    }
    link["dispheno"]={
        "st":2,
        "ed":36
    }
    link["phenodis"]={
        "st":2,
        "ed":36
    }
    link["gogene"]={
        "st":37,
        "ed":39
    }
    link["genego"]={
        "st":37,
        "ed":39
    }
    link["genepath"]={
        "st":40,
        "ed":40
    }
    link["pathgene"]={
        "st":40,
        "ed":40
    }
    link["genepro"]={
        "st":41,
        "ed":51
    }
    link["progene"]={
        "st":41,
        "ed":51
    }
    link["propro"]={
        "st":52,
        "ed":53
    }
    return link

def getfinal(src,st,ed,linkst,linked,dst,hop):
    candidatetop1=[]#pheno
    for i in range(100):
        candidatetop1.append({
            "score":1000000,
            "num":-1,
            "path":""
        })
    isdrug=0
    for lst in src:
        for i in range(dst,dst+1):
            for lik in range(linkst,linked):
                score=lst["score"]+get_score(entity_embedding[lst["num"]],relation_embedding[lik*2],drug_embedding[i])
                score=score/hop
                if score>candidatetop1[99]["score"]:
                    continue
                for disease in range(98,-1,-1):
                    if score<candidatetop1[disease]["score"]:
                        candidatetop1[disease+1]=candidatetop1[disease]
                    else:
                        candidatetop1[disease+1]={
                            "score":score,
                            "num":i,
                            "path":lst["path"]+"-"+str(i)
                        }
                        break
                    if disease==0:
                        candidatetop1[0]={
                            "score":score,
                            "num":i,
                            "path":lst["path"]+"-"+str(i)
                        }
                        break
    return candidatetop1

def getneighbor(src,st,ed,linkst,linked):
    candidatetop1=[]#pheno
    for i in range(100):
        candidatetop1.append({
            "score":1000000,
            "num":-1,
            "path":""
        })
    isdrug=0
    for lst in src:
        print(lst["num"])
        for i in range(st,ed):
            for lik in range(linkst,linked):
                score=0
                if isdrug==1:
                    score=lst["score"]+get_score(drug_embedding[lst["num"]],relation_embedding[lik*2],entity_embedding[i])
                else:
                    score=lst["score"]+get_score(entity_embedding[lst["num"]],relation_embedding[lik*2],entity_embedding[i])
                if score>candidatetop1[99]["score"]:
                    continue
                for disease in range(98,-1,-1):
                    if score<candidatetop1[disease]["score"]:
                        candidatetop1[disease+1]=candidatetop1[disease]
                    else:
                        candidatetop1[disease+1]={
                            "score":score,
                            "num":i,
                            "path":lst["path"]+"-"+str(i)
                        }
                        break
                    if disease==0:
                        candidatetop1[0]={
                            "score":score,
                            "num":i,
                            "path":lst["path"]+"-"+str(i)
                        }
                        break
    return candidatetop1
if __name__ == '__main__':
    schema=get_schema()
    link=get_link()
    fro=838
    dst=349
    originsrc=[]
    originsrc.append({
            "score":0,
            "num":fro,
            "path":str(fro)
        })
    schem=[["dis","pheno","dis"],["dis","gene","dis"],["dis","gene","path","gene","dis"],["dis","gene","go","gene","dis"],["dis","gene","pro","gene","dis"]]
    for sche in schem:
        src=originsrc
        for i in range(1,len(sche)):
            st=schema[sche[i]]["st"]
            ed=schema[sche[i]]["ed"]+1
            linkst=link[sche[i-1]+sche[i]]["st"]
            linked=link[sche[i-1]+sche[i]]["ed"]+1
            src=getneighbor(src,st,ed,linkst,linked)
        src=getfinal(src,0,8352,0,1,dst,len(sche)-1)
        print(sche)
        print(src)