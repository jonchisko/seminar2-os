# import pandas, datetime and numpy
import pandas as pd
from datetime import date
import numpy as np
import operator
import time
from numba import cuda


#Branje ocen (6)
class UserItemData:

    #konstruktor
    def __init__(self, path, **kwargs):
        #**kwargs, from_date, to_date, min_ratings
        #from_date, to_date input format day.month.year
        #min_ratings input format integer

        if path != "saved_user":
            self.data_frame = pd.read_csv(path, sep="\t")
            #dodaj stolpec date!
            #apliciramo funkcijo date na vsako vrstico -> axis=1, da dobimo na vrstice
            date_col = self.data_frame.loc[:,["date_year", "date_month", "date_day"]].apply(lambda row: date(row[0], row[1], row[2]), axis=1)
            #self.data_frame = pd.concat([self.data_frame, date_col], axis=1)
            self.data_frame = self.data_frame.assign(dated = date_col)


            if("start_date" in kwargs):
                date_arr_from = kwargs.get("start_date").split(".")
                #date(year, month, day)
                from_date = date(int(date_arr_from[-1]), int(date_arr_from[-2]), int(date_arr_from[-3]))
                self.data_frame = self.data_frame.loc[self.data_frame["dated"] > from_date, :]

            if("end_date" in kwargs):
                date_arr_to = kwargs.get("end_date").split(".")
                to_date = date(int(date_arr_to[-1]), int(date_arr_to[-2]), int(date_arr_to[-3]))
                self.data_frame = self.data_frame.loc[self.data_frame["dated"] < to_date, :]

            if("min_ratings" in kwargs):
                min_ratings = kwargs.get("min_ratings")
                #grupa določenega movieIDija mora biti večja od specificirane dolžine oz. števila ocen
                self.data_frame = self.data_frame.groupby("movieID").filter(lambda x: len(x) > min_ratings)


        else:
            UserItemData.read_data(self)

    #method how many movies were read
    def nratings(self):
        return self.data_frame.shape[0]

    #method save read data
    def save_data(self):
        import pickle
        #save file pickle
        self.data_frame.to_pickle(path="saved_user")

    def read_data(self):
        self.data_frame = pd.read_pickle(path="saved_user")

#TEST1
"""uim = UserItemData('data/user_ratedmovies.dat')
print("Prvi test:", uim.nratings())
uim = UserItemData('data/user_ratedmovies.dat', start_date = '12.1.2007', end_date='16.2.2008', min_ratings=100)
print("Drugi test:", uim.nratings())
uim.save_data()"""


#Branje filmov (6)
class MovieData:

    def __init__(self, path):
        if path != "saved_movie":
            self.data_frame = pd.read_csv(path, sep="\t", encoding="latin1")
        else:
            MovieData.read_data(self)

    def get_title(self, movieID):
        #return title of movie ID
        return self.data_frame.loc[self.data_frame["id"] == movieID, "title"].values[0] #.values[0], zato, da vrne le string title, drugače dobiš še vrstico in kateri tip je

    def save_data(self):
        self.data_frame.to_pickle(path="saved_movie")

    def read_data(self):
        self.data_frame = pd.read_pickle(path="saved_movie")
#TEST2
"""md = MovieData('data/movies.dat')
print(md.get_title(1))"""


"""
Prediktorji:
fit - učenje parametrov
predict - napovedovanje prediktorjev
"""

#Naključni prediktor (6)
class RandomPredictor:


    def __init__(self, min_rate, max_rate):
        #nakljucni prediktor, ki vrne oceno med min in max
        self.min_r = min_rate
        self.max_r = max_rate

    def fit(self, x):
        #kjer je x tipa UserItemData
        #nauci se na podatkih - ucnih, itak so random predikcije, zakaj bi rabil ucenje sploh?
        #grupiraj po filmih
        #česa se bom naučil pri random prediktorju ???
        self.user_data = x.data_frame

    def predict(self, user_id):
        #vrni predikcije filmov za user-ja
        #metoda predict pa vrne naključno vrednost (med min in max) za VSAK produkt.
        df = MovieData("saved_movie").data_frame
        predikcije = dict()
        for movieID in df["id"]:
            predikcije[movieID] = np.random.randint(self.min_r, self.max_r+1, 1)
        return predikcije


#TEST3
#md = MovieData('data/movies.dat')
#uim = UserItemData('data/user_ratedmovies.dat')

#uim.save_data()
#md.save_data()
"""
md = MovieData("saved_movie")
uim = UserItemData("saved_user")
rp = RandomPredictor(1, 5)
rp.fit(uim)
path = 'data/movies.dat'
pred = rp.predict(78)
print(type(pred))
items = [1, 3, 20, 50, 100]
for item in items:
    print("Film: {}, ocena: {}".format(md.get_title(item), pred[item]))"""

#Priporočanje (6)
class Recommender:

    def __init__(self, prediktor):
        self.prediktor = prediktor

    def fit(self, x):
        #kjer je x tipa UserItemData
        #za ucenje modela
        self.prediktor.fit(x)

    def recommend(self, userID, n=10, rec_seen=True):
        # vrne urejen seznam priporočenih produktov za uporabnika userID.
        # Parameter n določa število priporočenih filmov, z rec_seen pa določimo ali hočemo med priporočenimi tudi
        # že gledane (tiste, ki jim je uporabnik že dal oceno) ali ne."""
        ocene = self.prediktor.predict(userID)

        #sedaj imamo slovar s ključi, ki so movie ID in vrednostjo, ki je napoved
        user_data = self.prediktor.user_data
        #spremeni v list, drugače if x in videni_filmi ne deluje korektno, verjetno ker je to neka pandas "vrsta"
        videni_filmi = list(user_data.loc[user_data["userID"] == userID, "movieID"])
        if not rec_seen:
            # za kopiranje tistih, ki rabimo, če je rec_seen False
            final_grade = dict()
            for m_id in ocene:
                if m_id not in videni_filmi:
                    final_grade[m_id] = ocene[m_id]
            ocene = final_grade
        urejeno = sorted(ocene.items(), key=operator.itemgetter(1), reverse=True)
        return urejeno[:n]

    #NALOGA EVALUATE OCENA 7
    def evaluate(self, test_data, n):
        # test_data =  sprejme testne podatke
        # izračuna povprečne MAE, RMSE, priklic, natančnost, F1
        # Za priklic, natančnost in F1 boste morali za vsakega uporabnika izbrati nekaj priporočenih produktov. Jaz sem se odločil, da vzamem tiste,
        # ki jih je uporabnik ocenil bolje od svojega povprečja. Pri tem upoštevajte, da
        # ne priporočate že gledanih produktov in da parameter
        # n označuje število priporočenih produktov.

        #pridobi napovedi, ki si jih dobil iz učnih
        #podatki so ločeni po datumu na učne in testne, glej teste spodaj v kodi
        test_frame = test_data.data_frame
        #users - > spodnja koda vrže nepodvojene uporabinške IDije
        users = self.prediktor.user_data["userID"].unique()
        mse, mae, count, precision, recall, count2 = 0, 0, 0.000001, 0, 0, 0.000001
        for user_id in users:

            #produkti v testni množici, ki jih je uporabnik ocenil in jim je dal oceno, ki je višja od njegovega povprečja.
            # S tem določimo produkte, ki so uporabniku všeč. Te produkte rabimo, da lahko merimo precision in recall.

            user_data_test = test_frame.loc[test_frame["userID"] == user_id, ["movieID","rating"]]
            #predicikcije na učnih
            predikcija = Recommender.recommend(self, user_id, n=n, rec_seen=False) #to je seznam terk movie id in vrednost
            if(len(user_data_test) == 0 or len(predikcija) == 0):
                #ta user očitno ni ocenil nobenega filma po 2008
                #oziroma v drugem primeru, ta user je ocenil vse filme, ki mu jih recommendamo in se zato vrne prazno
                continue
            #za precizijo, priklic in f vzemi le n
            ppf = predikcija
            #priporcam tiste vecje od uporabnikovega povprecja
            avg = np.mean([value for id, value in ppf]) #avg napovedana ocena
            priporocam = [id for id,value in ppf if value >= avg] #if value >= avg]
            #Start
            #iz testne pa zdaj vzames tudi le tiste, ki so visji od povprecja :)
            avg2 = np.mean(user_data_test["movieID"].values)
            v = [x for x in user_data_test["movieID"].values if x >= avg] #avg naucen iz ucnih vzamemo?
            #End
            presek = len(np.intersect1d(priporocam, v)) #kaj priporocas in kaj je user pogledal
            precision += presek / len(priporocam)
            recall += presek / len(user_data_test["movieID"])
            count2 += 1
            #primerjaj predikcije le po tistih, ki jih ima uporabnik zdaj ocenjene
            ocenjeni = user_data_test["movieID"].values
            for movie_id, value in predikcija:
                #poglej, ce je ta movie_id sedaj ocenjen!
                if movie_id in ocenjeni:
                    mse += (value - user_data_test.loc[user_data_test["movieID"] == movie_id, "rating"].item())**2
                    mae += np.abs(value - user_data_test.loc[user_data_test["movieID"] == movie_id, "rating"].item())
                    count+=1

        return (mse/count, mae/count, precision/count2, recall/count2, 2*precision*recall/(precision+recall+0.000001) * 1/count2)



#TEST4
"""
md = MovieData("saved_movie")
uim = UserItemData("saved_user")
rp = RandomPredictor(1, 5)
rec = Recommender(rp)
rec.fit(uim)
rec_items = rec.recommend(78,n = 5, rec_seen=False)
for idmovie, val in rec_items:
    print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))"""

#Napovedovanje s povprečjem (6)
class AveragePredictor:

    def __init__(self, b=0):
        # je parameter formule za povprečje. Če je b=0, gre za navadno povprečje
        self.b = b

    def fit(self, x):
        self.user_data = x.data_frame
        #nastavi parametre
        #VS je vsota vseh ocen za ta film
        #n je stevilo ocen, ki jih je ta film dobil
        #g_avg je povprecje cez vse
        self.filmi_weighted_avg = dict()
        g_avg = np.mean(self.user_data["rating"])
        grouped = self.user_data.groupby("movieID")
        vs = grouped["rating"].agg(np.sum)
        for name, group in grouped:
            n = len(group)
            if not n:
                self.filmi_weighted_avg[name] = 0
            else:
                #name uporabimo, name je namrec kar movieID, ker smo po temu grupirali
                self.filmi_weighted_avg[name] = (vs[name] + self.b * g_avg) / (n + self.b)

    def predict(self, user_id):
        #ker v recommenderju poskrbimo za to, ali damo ven tudi filme, ki jih je user gledal ali ne
        #tukaj lej vrnemo cel dict
        return self.filmi_weighted_avg

#TEST5
"""md = MovieData("saved_movie")
uim = UserItemData("saved_user")
rec = Recommender(AveragePredictor(b=100))
rec.fit(uim)
rec_items = rec.recommend(78, n = 5, rec_seen=False)
for idmovie, val in rec_items:
    print("Id: {}, Film: {}, ocena: {}".format(idmovie, md.get_title(idmovie), val))"""

#Priporočanje najbolj gledanih filmov (6)
class ViewsPredictor:

    def __init__(self):
    #za vsak film vrne število ogledov posameznega filma. To je priporočanje najbolj gledano
        self.film_gledano = dict()

    def fit(self, x):
        self.user_data = x.data_frame
        #grupiram po filmih
        grouped = x.data_frame.groupby("movieID")
        for name, x in grouped:
            #"ocena" je kar velikost grupe filma oziroma stevilo ocen
            self.film_gledano[name] = len(x)

    def predict(self, user_id):
        return self.film_gledano

#TEST6
"""md = MovieData("saved_movie")
uim = UserItemData("saved_user")
rec = Recommender(ViewsPredictor())
rec.fit(uim)
rec_items = rec.recommend(78, n = 5, rec_seen=False)
for idmovie, val in rec_items:
    print("Id: {}, Film: {}, ocena: {}".format(idmovie, md.get_title(idmovie), val))"""


#Priporočanje kontroverznih filmov (7)
"""Kako bi ocenili kontroverznost (produkti, ki imajo veliko dobrih in veliko slabih ocen) ? Ali je "nepersonaliziran" 
način priporočanja primeren za take produkte? 
Napišite prediktor za najbolj kontroverzne produkte, kjer je film lahko kontroverzen, če ima vsaj n ocen. 
Za mero uporabite standardno deviacijo ocen. Če mora film imeti vsaj 100 ocen, dobimo: """
"""
ODGOVOR:
Kontroverznost produktov bi gledal glede na razpršenost ocen, ki so mu bile dane. Najbolj kontroverzni produkti bodo imeli največjo razpršenost ocen - torej 1 in 5 v našem primeru.
To pa moramo še utežiti oziroma uporabiti le filme, ki imajo dovolj ocen, saj bi lahko v primeru neupoštevanja tega dobil film z dvema ocenama (1 in 5), maksimalno razpršenost in 
bi bil zelo kontroverzen, čeprav je verjetnost tega majhna, saj je premalo ocen.

Ne vem kaj točno vprašanje sprašuje.
"""
class STDPredictor:

    def __init__(self, n):
        self.min_num = n

    def fit(self, x):
        self.user_data = x.data_frame
        #grupiram po filmih
        grouped = self.user_data.groupby("movieID")
        self.dict_predikcij = dict()
        for name, group in grouped:
            if len(group)>self.min_num:
                #ce je grupa filma vecja od min, potem aggregiraj standardno diviacijo po ocenah od grupe tega filma
                self.dict_predikcij[name] = group["rating"].agg(np.std)

    def predict(self, user_id):
        return self.dict_predikcij

#TEST7
"""md = MovieData("saved_movie")
uim = UserItemData("saved_user")
rp = STDPredictor(100)
rec = Recommender(rp)
rec.fit(uim)
rec_items = rec.recommend(78, n=5, rec_seen=False)
for idmovie, val in rec_items:
    print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))"""





######## 2. VAJE #########

#Napovedovanje ocen s podobnostjo med produkti (6)

class ItemBasedPredictor:

    def __init__(self, min_values=0, threshold=0):
        self.mv = min_values
        self.th = threshold

    def fit(self, x):
        self.user_data = x.data_frame

        #dict filmov, kjer je kljuc film id, vrednost pa slovar kljuc film id in podobnost
        self.dict_sim = dict()

        #self.movie_data = MovieData("saved_movie").data_frame

        #user data je zdesetkan na le tiste vrstice, kjer so filmi imeli več kot 1000 ocen
        #grupiram po movieID
        #g = [id for id, _ in self.user_data.groupby("movieID")]
        g = list(self.user_data.groupby("movieID").apply(lambda x: x.name)) #spremenim v list, ker drugace je series in je g[indeks] v bistvu g[ključ] in ne indeks
        #^ nič hitreje, tudi če ne spremenim v list in uporabim drugačen način zanke spodaj
        #Da bi naredil matriko stevilo filmov x stevilo filmov, bi to bili v tem primeru pri filmih z nad 1000 ocenami, 81x81
        #to bi bilo okoli 6000 računanj, vsakega z vsakim, a ker vem, da je par a b isto kot par b a, lahko računam le "trikotniško"
        #število se zmanjša na: prvo 81, nato drugi film ni treba s prvim zato 80, nat 79 et cetera
        #to pa je število operacij = n(n+1)/2
        #v mojem primeru okoli 3000, prostorsko pa bom vseeno šel na 2x in zasedel še spodnjo matriko
        self.movie_panda = pd.DataFrame(np.zeros((len(g), len(g))), index=g, columns=g)

        self.pari_podobnosti = []

        for i in range(len(g)):
            for j in range(i+1, len(g)):
                self.movie_panda.iloc[i, j] = ItemBasedPredictor.similarity(self, g[i], g[j])
                #še spodnjo diagonalo zapolnim, mi bo lažje potem v predict, ne bo časovno računanje, ker je že poračunan
                self.movie_panda.iloc[j, i] = self.movie_panda.iloc[i, j]

                #dodam še v par podobnosti
                #self.pari_podobnosti[ (g[i], g[j]) ] = self.movie_panda.iloc[i, j]
                self.pari_podobnosti.append( (g[i], g[j], self.movie_panda.iloc[i, j]))
        #v bistvu podobno kot zgoraj, po matriki uporabim funkcijo similartiy, ki prejme element in vstavi njegovo vrstico id in stolpec id v izračun.
        #self.movie_panda = self.movie_panda.apply(lambda row: ItemBasedPredictor.similarity(self, row.index, row.name), axis=1)
        """      0
            1 1   1
              2   2
              3   3
            2 1   2
              2  10
              3   6
            3 1   3
              2   6
              3   9
              name[0] = 1, name[1] = 1 vrstica 1
              name[0] = 1, name[1] = 2 vrstica 1
              name[0]= 1, name[1] = 3, vrstica 1
              name[0] =2, name[1] = 1, vrstica 2
              name[0]= 2, name[1] = 2, vrstica 2
              itd.
              prvi stolpec so indeksi vrstic
              drugi stolpec so names oz stolpci, 
              zadnji stolpec so vrednosti
              tako dobim vsak indeks vrstice z vsakim stolpcem
              in element wise, sam pač je počasneje od double fora v bistvu... vsaj v mojem primeru ???
              """
        #self.movie_panda = self.movie_panda.stack().to_frame().apply(lambda x: ItemBasedPredictor.similarity(self, x.name[0], x.name[1]), axis=1).unstack() #počasneje

    def predict(self, user_id):
        #formula,
        #pred(user5, produkt6) = vsota_od vseh itemsov ( podobnost(item, produkt6) * rating od item ) uteženo z vsoto podobnosti med item in p6
        #skratka gremo čez vse produkte, ki jih je ocenil user5. Bolj kot je item podoben produkt6, bolj velja ocena, ki jo je dal user5 temu itemu.
        #samo da tukaj ne vrnes le da produkt6, ampak za vse produkte, ki jih ta user še ni gledal -> oziroma napovej kar za vse filme, selekcijo nato itak delaš
        #v recommender
        napovedi = dict()
        ocenjeni = self.user_data.loc[self.user_data["userID"] == user_id, ["movieID", "rating"]]
        vsi_filmi = self.movie_panda.index
        for film in vsi_filmi:
            #izberemo prvo vrstice, ki vsebujejo filme, ki jih je ocenil uporabnik
            #nato izberemo le tisti stolpec, za kateri film trenutno računamo, tako dobimo v selekciji array podobnosti, to so utezi
            selekcija = self.movie_panda[self.movie_panda.index.isin(ocenjeni["movieID"])][film] #prvo izberemo le vrstice, filmi, ki so bili ocenjeni, nato pa le stolpec filma, ki ga "napovedujemo"
            #->ce ne bi delal z boolean, bi takole slo [stolpec_ime][vrstica_ime]
            #tole bo verjetno vedno delalo, saj bo isin šel po vrsti po matriki gledat in bodo filmi vedno od manjsega k vecjemu indeksu, ravno tako bodo v ocenjeni[rating], tako da
            #se bodo vedno pravilno zmnozili istolezni(utez in ocena)

            zgornja_vsota = np.sum(np.array(selekcija)*np.array(ocenjeni["rating"]))
            spodnja_vsota = np.sum(np.array(selekcija))
            if spodnja_vsota == 0:
                #vsota bo nic takrat ko bo selekcija le en film, in bo film zanka prisel do tega istega filma (:
                napovedi[film] = 0
            else:
                napovedi[film] = zgornja_vsota/spodnja_vsota

        return napovedi


    def return_podobne(self):
        #vrni najbolj podobne
        #sortirano = sorted(self.pari_podobnosti.items(), key=operator.itemgetter(1), reverse=True)
        sortirano = sorted(self.pari_podobnosti, key=lambda x: x[2], reverse=True)
        return sortirano

    def similarItems(self, item, n):
        #Kaj bi pokazali v kategoriji "Gledalci, ki so gledali A, so gledali tudi B"?
        #vrne n najbolj podobnih filmov izbranemu filmu.
        #i je index oziroma movieId, self.movie_panda[item][i] je pa stolpec od izbranega filma [i] pa movieId vrstica od filma s katerimi primerjamo - skratka podobnost je to
        podobni_izbranemu = [ (i, self.movie_panda[item][i]) for i in self.movie_panda[item].index] #izberemo stolpec, ki ima item ime, to dobimo zdej vrsto podobnosti
        return sorted(podobni_izbranemu, key=lambda x: x[1], reverse=True)[:n] #vrni le n najbolj


    def similarity(self, p1, p2):
        #Podobnost izračunajte s popravljeno cosinusno razdaljo
        #Če je izračunana podobnost med produktoma manjša od threshold ali če imamo manj
        #kot min_values uporabnikov, ki je ocenilo oba filma, naj bo podobnost med produktoma 0
        #self.user_data = uim.data_frame
        #pridobi vektorje, p1 in p2 sta enako dolga, najdi le skupne uporabnike in jih zlozi istolezno
        first_data = self.user_data.loc[self.user_data["movieID"] == p1, ["userID", "rating"]]
        second_data = self.user_data.loc[self.user_data["movieID"] == p2, ["userID", "rating"]]

        #presek userjev, ki so v obeh
        users = np.intersect1d(first_data["userID"], second_data["userID"])

        if len(users) <= self.mv:
            return 0

        #dobim skupine userjev, ki so ocenili oba produkta. So zdruzene vrstice po userjih, torej, ce je user1 ocenil oba filma, je tukaj notri v svoji skupini in predstavlja n vrstic
        grupe_user = self.user_data.groupby("userID")
        #izracunam povprecje na skupino(user), njegova povprecna ocena
        povprecje_user = grupe_user["rating"].agg(np.mean)

        #ker ni uporabljen iloc, je to vbistvu po imenu vrstice, in to je indeks!!!
        #print(povprecje_user[78])
        # filtriram tiste userje, ki niso prisotni v obeh filmih
        #dam se v numpy, ker ce odstejem pandas series od series dobim napacen rezultat
        povprecje_user = np.array(povprecje_user[povprecje_user.index.isin(users)])

        #vektorji
        #avg_user = np.mean(self.user_data.loc[self.user_data["userID"].isin(users), "rating"])
        # ne pozabi odšteti user bias-a od ratinga, torej odštej od ratinga povprečno user oceno
        p1v = first_data.loc[first_data["userID"].isin(users), "rating"] - povprecje_user   #ce bi bil scalar, sepravi da bi selecta eno vrednost, bi vrednost izluscil z .item()
        p2v = second_data.loc[second_data["userID"].isin(users), "rating"] - povprecje_user

        #dot
        p1p2 = np.dot(np.array(p1v), np.array(p2v))
        #dolzini
        lengthp1 = np.sqrt(np.dot(p1v, p1v))
        lengthp2 = np.sqrt(np.dot(p2v, p2v))
        sim = p1p2 / (lengthp1 * lengthp2)
        if sim < self.th:
            return 0
        return sim

#TEST8

"""md = MovieData('saved_movie')
uim = UserItemData('saved_user')
rp = ItemBasedPredictor()
rec = Recommender(rp)

t1 = time.time()
#rec.fit(uim)
print(time.time()-t1)

print("Podobnost med filmoma 'Men in black'(1580) in 'Ghostbusters'(2716): ", rp.similarity(1580, 2716))
print("Podobnost med filmoma 'Men in black'(1580) in 'Schindler's List'(527): ", rp.similarity(1580, 527))
print("Podobnost med filmoma 'Men in black'(1580) in 'Independence day'(780): ", rp.similarity(1580, 780))

print("\n")
rec_items = rec.recommend(78, n=15, rec_seen=False)
for idmovie, val in rec_items:
    print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))


najbolj_podobni = rp.return_podobne()
print("\n")
for par1, par2, value in najbolj_podobni[:20]:
    print("Film 1: %s, Film2: %s, podobnost: %f" % (md.get_title(par1), md.get_title(par2), value))

rec_items = rp.similarItems(4993, 10)
print('\nFilmi podobni "The Lord of the Rings: The Fellowship of the Ring": ')
for idmovie, val in rec_items:
    print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))"""


######################
# Priporočilo zase (6)
# Naredite še priporočilo zase z metodo "item-based"; izberite si cca. 20 filmov,
# ki jih poznate in jih ročno ocenite. Dodajte svoje ocene v movielens bazo in si priporočite 10 filmov.




md = MovieData('saved_movie')
uim = UserItemData('saved_user')
#userID	movieID	rating	date_day	date_month	date_year	date_hour	date_minute	date_second
#75	        3	   1	   29	         10	        2006	   23	        17	        16
moje_ocene = pd.DataFrame([[123456789, 5952, 5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 7153, 5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 4993, 5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 586, 2, 25, 12, 2017, 10, 10, 10],
                           [123456789, 6874, 3.5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 260, 5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 1196, 4.5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 344, 1, 25, 12, 2017, 10, 10, 10],
                           [123456789, 367, 1, 25, 12, 2017, 10, 10, 10],
                           [123456789, 1210, 3.5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 2628, 2.5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 780, 2, 25, 12, 2017, 10, 10, 10],
                           [123456789, 2571, 5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 541, 4, 25, 12, 2017, 10, 10, 10],
                           [123456789, 527, 5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 1136, 5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 2, 2.5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 1, 3.5, 25, 12, 2017, 10, 10, 10],
                           [123456789, 48, 3, 25, 12, 2017, 10, 10, 10],
                           [123456789, 364, 4.5, 25, 12, 2017, 10, 10, 10]])
#te indeksi so col namesi, zato so ubistvu kr isti ..... wtf lol ta row je kr series al kaj, mogoče .iloc drgač vrne kokr če po .loc
moje_ocene = moje_ocene.assign(dated=moje_ocene.iloc[:,[5, 4, 3]].apply(lambda row: date(row[5], row[4], row[3]), axis=1))
moje_ocene.columns = uim.data_frame.columns
moje_ocene = moje_ocene.sort_values(["movieID"])

#ignorira indekse da se ne ponavljajo, zdruzim dataframea
uim.data_frame = uim.data_frame.append(moje_ocene, ignore_index=True)


"""rp = ItemBasedPredictor()
rec = Recommender(rp)

t1 = time.time()
rec.fit(uim)
print("\n")
print(time.time()-t1)
print("\n")
rec_items = rec.recommend(123456789, n=10, rec_seen=False)
for idmovie, val in rec_items:
    print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))"""




###################################
###################################
###################################


class SlopeOnePredictor:

    # Izbereš i film in j film
    # Greš sum po uprabnikih:
    # izbereš uporabnika in če ima oba filma ocenjena, izračunaš razliko
    # greš na naslednjega uporabnika in storiš isto
    # -> dobljeni sum še normaliziraš s številom uporabnikov, ki so ocenili oba filma
    # -> DIF i,j  ... to je "razlika" za en par

    def __init__(self):
        pass

    def fit(self, x):
        #user data
        self.user_data = x.data_frame
        #SESTAVIM MATRIKO MOVIE BY MOVIE PO ČISTO ENAKI LOGIKI KOT V ITEMBASED
        g = list(self.user_data.groupby("movieID").apply(lambda x: x.name)) #spremenim v list, ker drugace je series in je g[indeks] v bistvu g[ključ] in ne indeks

        self.movie_panda = pd.DataFrame(np.zeros((len(g), len(g))), index=g, columns=g) #dev za vsak par film
        self.users_panda = pd.DataFrame(np.zeros((len(g), len(g))), index=g, columns=g) #stevilo uporabnikov, ki so ocenili ta par filmov
        for i in range(len(g)):
            for j in range(i+1, len(g)):
                self.movie_panda.iloc[i, j], self.users_panda.iloc[i, j] = SlopeOnePredictor.dif(self, g[i], g[j])
                #še spodnjo diagonalo zapolnim, mi bo lažje potem v predict, ne bo časovno računanje, ker je že poračunan
                #DEV skopiramo v spodnji trikotnik matrike z minus predznakom. DEV nam namreč pove koliko je par1 boljsi od par2, torej ko kopiras v polje kjer je par2,par1 in ne par1,par2
                #moras dati nasprotni predznak
                self.movie_panda.iloc[j, i], self.users_panda.iloc[j, i] = -self.movie_panda.iloc[i, j], self.users_panda.iloc[i, j]

    def predict(self, user_id):

        napovedi = dict()
        #user je ocenil tele
        ocenjeni = self.user_data.loc[self.user_data["userID"] == user_id, ["movieID","rating"]]
        #vsi filmi, ki sploh so
        vsi_filmi = self.movie_panda.index
        for film in vsi_filmi:

            dev_ocenjeni_film = self.movie_panda[film][ocenjeni["movieID"]]

            st_ocen_user = self.users_panda[film][ocenjeni["movieID"]]
            #če bo user mel le eno oceno, bo ta pri film = ocena filma
            #np.sum(st_ocen_user) == 0, kar bo division err
            if np.sum(st_ocen_user) == 0:
                napovedi[film] = 0
            else:
                napovedi[film] = np.sum((ocenjeni["rating"].values - dev_ocenjeni_film.values) * st_ocen_user.values) / np.sum(st_ocen_user.values)
        return napovedi

    def dif(self, j, i):
        #arg j je film j
        #arg i je film i

        #selekcioniraj ven userje, ki so ocenili j in i
        #>skupina po userID, v vsaki skupini obdrzim tiste, ki imajo notri j in i
        df_f = self.user_data.loc[np.logical_or(self.user_data["movieID"] == i, self.user_data["movieID"] == j), ["userID", "rating", "movieID"]]
        #če imaš in I in J, potem sta dva vnosa v df_f in nsi duplikat
        #df_uniq vsebuje le ne duplicirane, torej une userje, ki imajo ocenjen le bodisi i ali bodisi j
        df_uniq = df_f.drop_duplicates(subset="userID", keep=False)
        # se je userid iz org df_f v uniq, potem ma ta le en vnos in takih nočemo
        df_f = df_f.loc[np.logical_not(df_f["userID"].isin(df_uniq["userID"])), :]

        #^tole je tko 100kratna pohitritev vsaj v primerjavi z filtriranjem groupby objetka

        #ce applyam takole filter funkcijo, traja tole namesto skoraj 0, 1/3 sekunde, kar je absolutno preveč tudi za množico filmov z vsaj 1000 ocenami
        #groupd_users_df = self.user_data.groupby("userID").filter(lambda x: j in x["movieID"].values and i in x["movieID"].values)

        #groupd_users_df = df_f.groupby("userID")#.filter(lambda x: len(x)==2) #tako sem selekcioniral prej data, da ce bo grupa bila dolzine 2, bo imela i oceno in j oceno, drugace ne
        #to je sedaj 3x hitreje kot prejsnji nacin filtra

        #print([len(g) for _,g in groupd_users_df])


        # izberem ocene userjev za j
        users_df_j = df_f.loc[df_f["movieID"] == j, "rating"]
        #izberem ocene userjev za i
        users_df_i = df_f.loc[df_f["movieID"] == i, "rating"]
        #koncni izracun sum po userjih razlik ocen za j in i, nato pa sum normaliziran s stevilom userjev, ki je dalo ocene obema
        dev = np.sum((users_df_j.values - users_df_i.values))/len(users_df_j)
        #vrni dev vrednosti in stevilo uporabnikov, ki so ocenil i in j
        return (dev, len(users_df_j))


#TEST

"""md = MovieData('saved_movie')
uim = UserItemData('saved_user')
rp = SlopeOnePredictor()
rec = Recommender(rp)
rec.fit(uim)

#rp.dif(32, 110)

print("Predictions for 78: ")
rec_items = rec.recommend(78, n=15, rec_seen=False)
for idmovie, val in rec_items:
    print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))"""



class HybridPredictor:

    #uporabi slopeone, itembased in  najbolje ocenjeno - average predictor???

    def __init__(self, min_values=0, threshold=0, b=0):
        self.mv = min_values
        self.th = threshold
        self.b = b

        self.item_based = ItemBasedPredictor(self.mv, self.th)
        self.slope = SlopeOnePredictor()
        self.avg = AveragePredictor(self.b)

    def fit(self, x):
        self.user_data = x.data_frame
        self.item_based.fit(x)
        self.slope.fit(x)
        self.avg.fit(x)

    def predict(self, user_id):

        pred_ib = self.item_based.predict(user_id)
        pred_s = self.slope.predict(user_id)
        pred_avg = self.avg.predict(user_id)

        napovedi_final = dict()
        #le navadno povprecje vseh napovedi
        for film_id in pred_ib:
            napovedi_final[film_id] = (pred_ib[film_id] + pred_s[film_id] + pred_avg[film_id])/3
        return napovedi_final

#TEST
"""print("\nTest hibrid")
md = MovieData('saved_movie')
uim = UserItemData('saved_user')
rp = HybridPredictor()
rec = Recommender(rp)
rec.fit(uim)

#rp.dif(32, 110)

print("Predictions for 78: ")
rec_items = rec.recommend(78, n=15, rec_seen=False)
for idmovie, val in rec_items:
    print("Film: {}, ocena: {}".format(md.get_title(idmovie), val))"""








#TEST RECOMMENDER EVALUATE METHOD
#md = MovieData('data/movies.dat')
#uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000, end_date='1.1.2008')
#uim_test = UserItemData('data/user_ratedmovies.dat', min_ratings=200, start_date='2.1.2008')

#SLOPE
#rp = SlopeOnePredictor()
#rec = Recommender(rp)
#rec.fit(uim)
#time1 = time.time()
#mse, mae, prec, rec, f = rec.evaluate(uim_test, 20)
#print("time:", time.time()-time1)
#print("SLOPE ONE", mse, mae, prec, rec, f)


#ITEM BASED
"""rp = ItemBasedPredictor()
rec = Recommender(rp)
rec.fit(uim)
#mse, mae, precision, recall, f = rec.evaluate(uim_test, 20)
time1 = time.time()
mse, mae, prec, rec, f = rec.evaluate(uim_test, 20)
print("time:", time.time()-time1)
print("Item Based predictor", mse, mae, prec, rec, f)


#HybridPredictor
rp = HybridPredictor()
rec = Recommender(rp)
rec.fit(uim)
#mse, mae, precision, recall, f = rec.evaluate(uim_test, 20)
time1 = time.time()
mse, mae, prec, rec, f = rec.evaluate(uim_test, 20)
print("time:", time.time()-time1)
print("Hybrid", mse, mae, prec, rec, f)"""


#time: 50.384504079818726
#SLOPE ONE 0.737326152762 0.631210888812 0.18542510880173224 0.08135736372584979 0.1130936217642868
#time: 19.427669286727905
#Item Based predictor 1.01097199952 0.697650666469 0.2000134266029619 0.08813128088987293 0.12235129796461698
#time: 68.26756930351257
#Hybrid 0.684466698395 0.611345915174 0.19345986304860757 0.0871357496992999 0.12015348378671341
#BEST
#Hybrid 0.684466698395 0.611345915174 0.19345986304860757 0.0871357496992999 0.12015348378671341







#Povezovalna pravila (8)

""" ČE ti je všeč film X in če ti je všeč film Y, POTEM ti bo zelo verjetno všeč tudi film Z.
Ker so povezovalna pravila namenjena podatkom z implicitnimi ocenami (dogodki), a pri filmih 
imamo eksplicitne vrednosti, moramo le-te najprej spremeniti v dogodke. Predlagam, da za dogodek označimo 
vse ocene višje od uporabnikovega povprečja. """


#uim = UserItemData('data/user_ratedmovies.dat')
#movies = MovieData('data/movies.dat')
#uim.save_data()
#movies.save_data()
#print("done")
#Moram narediti stolpce za film ID in ? tam kjer stolpec nima vrednosti in 1 tam kjer jo ima
#Vsaka vrstica so userji

data_frame = UserItemData('saved_user').data_frame
mobject = MovieData('saved_movie')
movie_frame = mobject.data_frame

user = data_frame.groupby("userID")
avg_user = user["rating"].mean() #average ocena za vsakega userja
indeksi = list(data_frame["userID"].unique()) #user IDs, zaradi lepsega v list


def asocpravila():
    #filmi ki so ocenjeni
    film_id = data_frame.groupby("movieID")
    columns_names = list(film_id.groups.keys()) #imena grup, prvo dobim od grup kljuce oz idije, nato pretvorim v list

    #ker je filmov okoli 10k, njhivoi indeksi pa gredo do 60k, je smiselno, da pretvorim ta njihov id v indeks do 10k in tega uporabim v matriki
    def movie_id2indeks(movie_id):
        return movie_frame.loc[movie_frame["id"].isin(movie_id), "id"].index.values #vrne indeks

    print("indeks filam s tem idijem 65133, to +1 je tudi stevilo stolpecv ", movie_id2indeks([65133]))
    maksimalni = np.max(columns_names)
    st_stolpcev = movie_id2indeks([maksimalni]) +1

    matrika = pd.DataFrame(np.zeros((len(indeksi), st_stolpcev[0])), index=indeksi, columns=movie_frame["id"])   #inekdsi len so vrstica, len columnsi so stolpci
    vrstica_filmi = dict() #ne bi bilo potrebno dicta vzet (: sam sej ni pomembn vrstni red, tako da vseeno, vazno je le da filme naprave

    for u_i in indeksi:
        vrstica_filmi[u_i] = data_frame.loc[np.logical_and(data_frame["userID"] == u_i, data_frame["rating"] > avg_user[u_i]), "movieID"].values  #tam kjer se user id ujema in da je rating večji od avg user ratinga, vzemi ven
        #seznamsek vrednosti film idov
    t1 = time.time()
    for i, u_i in enumerate(vrstica_filmi):
        # TUKAJ MATRIKA.ILOC, NASTAVIM TISTE STOLPCE NA 1, KJER SO FILMI , to ubistvu tukaj deluje kot indeks, vsak film je svoj indeks. ker pa je film indeks mnogo vecji od
        #filmi = np.apply_along_axis(movie_id2indeks, axis=0, arr=vrstica_filmi[u_i]) #dam vrstico filmov na u_i in ta vrne array id indeksov, skratka vrstica filmov se preslika v vrstico film indeksov
        filmi = vrstica_filmi[u_i]
        matrika.loc[u_i, filmi] = 1 #nastavim na gledano

    print(time.time()-t1)
    print("toystory number of ones", np.sum(matrika.iloc[:, 0]))
    print("Pred", matrika.shape)

    for col in matrika.columns:
        if matrika[col].sum()/matrika.shape[0] < 0.3:
            matrika = matrika.drop(col, 1)
    matrika[matrika==0] = np.nan
    print("Po removu samih praznih?", matrika.shape)
    matrika.to_csv("/users/js5891/Desktop/matrika.csv")

    #Izpišite pravilo z največjo podporo, zaupanjem in dvigom.


#asocpravila()
def rezultat():

    print(mobject.get_title(4993)+" -> "+mobject.get_title(5952))
    print("Supp:", 0.522, "Conf:",0.879, "Lift:",1.567)
    print(mobject.get_title(7153)+" -> "+mobject.get_title(5952))
    print("Supp:", 0.502, "Conf:",0.895, "Lift:",1.610)

#rezultat()

"""
The apriori principle can reduce the number of itemsets we need to examine. Put simply, the apriori principle states that if an itemset is infrequent, 
then all its subsets must also be infrequent. This means that if {beer} was found to be infrequent, we can expect {beer, pizza} to be equally or even more infrequent. 
So in consolidating the list of popular itemsets, we need not consider {beer, pizza}, nor any other itemset configuration that contains beer.
"""









# VIZUALIZACIJA DISTANC
# * MATRIKA RAZDALJ MED FILMI
#Če so naši elementi filmi, potem so to razdalje med filmi, ki jih lahko izračunamo npr. kot 1 - kosinusna podobnost
# potem daj v orange in shrani par vizualizacij ;)


class visualDistance:

    def __init__(self, df, mf):
        self.user_data = df
        self.mf = mf

    def matrikaRazdalj(self):

        # vzemi le filme, ki imajo več kot 1000 ocen
        grouped = list(self.user_data.groupby("movieID").apply(lambda x: x.name)) #get names

        #make matrix
        film_matrika = pd.DataFrame(np.zeros((len(grouped), len(grouped))))
        for i in range(len(grouped)):

            for j in range(len(grouped)):
                podobnost = self.podobonost(grouped[i], grouped[j])
                film_matrika.iloc[i, j] = 1 - podobnost
                film_matrika.iloc[j, i] = 1 - podobnost

        return film_matrika

    def podobonost(self, p1, p2):
        #Podobnost izračunajte s popravljeno cosinusno razdaljo
        #Če je izračunana podobnost med produktoma manjša od threshold ali če imamo manj
        #kot min_values uporabnikov, ki je ocenilo oba filma, naj bo podobnost med produktoma 0
        #self.user_data = uim.data_frame
        #pridobi vektorje, p1 in p2 sta enako dolga, najdi le skupne uporabnike in jih zlozi istolezno
        first_data = self.user_data.loc[self.user_data["movieID"] == p1, ["userID", "rating"]]
        second_data = self.user_data.loc[self.user_data["movieID"] == p2, ["userID", "rating"]]

        #presek userjev, ki so v obeh
        users = np.intersect1d(first_data["userID"], second_data["userID"])

        if len(users) <= 0:
            return 0

        #dobim skupine userjev, ki so ocenili oba produkta. So zdruzene vrstice po userjih, torej, ce je user1 ocenil oba filma, je tukaj notri v svoji skupini in predstavlja n vrstic
        grupe_user = self.user_data.groupby("userID")
        #izracunam povprecje na skupino(user), njegova povprecna ocena
        povprecje_user = grupe_user["rating"].agg(np.mean)

        #ker ni uporabljen iloc, je to vbistvu po imenu vrstice, in to je indeks!!!
        #print(povprecje_user[78])
        # filtriram tiste userje, ki niso prisotni v obeh filmih
        #dam se v numpy, ker ce odstejem pandas series od series dobim napacen rezultat
        povprecje_user = np.array(povprecje_user[povprecje_user.index.isin(users)])

        #vektorji
        #avg_user = np.mean(self.user_data.loc[self.user_data["userID"].isin(users), "rating"])
        # ne pozabi odšteti user bias-a od ratinga, torej odštej od ratinga povprečno user oceno
        p1v = first_data.loc[first_data["userID"].isin(users), "rating"] - povprecje_user   #ce bi bil scalar, sepravi da bi selecta eno vrednost, bi vrednost izluscil z .item()
        p2v = second_data.loc[second_data["userID"].isin(users), "rating"] - povprecje_user

        #dot
        p1p2 = np.dot(np.array(p1v), np.array(p2v))
        #dolzini
        lengthp1 = np.sqrt(np.dot(p1v, p1v))
        lengthp2 = np.sqrt(np.dot(p2v, p2v))
        sim = p1p2 / (lengthp1 * lengthp2)
        if sim < 0:
            return 0
        return sim

    def save2orange(self):
        import Orange.misc.distmatrix

        matrika = self.matrikaRazdalj()

        o = Orange.misc.distmatrix.DistMatrix(matrika)
        o.save("/users/js5891/Desktop/orange_razd.dst")

        matrika.to_csv("/users/js5891/Desktop/film_matrika.dst")


#df = UserItemData("data/user_ratedmovies.dat", min_ratings=1000).data_frame
#mf = MovieData("data/movies.dat").data_frame

#df = UserItemData('saved_user').data_frame
#mf= MovieData('saved_movie').data_frame


#vd = visualDistance(df, mf)
#vd.save2orange()








"""
1)
 *Inkrementalno testiranje in prečno preverjanje (cross-validation)

Namesto enkratnega testiranja je bolje, če razdelitev in ocenjevanje večkrat ponovimo. Pri prečnem preverjanju 
vse ocene razdelimo v nekaj delov (angl. fold). Npr. predpostavimo, da ocene razdelimo na deset delov. Potem desetkrat 
ponovimo učenje na devetih delih in testiranje na desetem (vsakič drugem). Končne vrednosti 
statistik so povprečne vrednosti čez posamezna testiranja.

Inkrementalno testiranje je najboljši približek delovanja realnega sistema, kadar imamo na voljo datume dogodkov (npr. ocen). 
Najprej izberemo začetni datum in se učimo samo na ocenah do tega datuma. Ta sistem testiramo na nekem oknu, npr. ocenah, 
ki so bile oddane v naslednjem tednu. Potem vključimo te ocene v učno množico in testiramo na tednu, ki sledi. To ponavljamo, 
dokler ne zmanjka ocen. Na koncu rezultate povprečimo. """

#rabil bom evaluate metodo
def predikt(uim, uim_test):
    rp = ItemBasedPredictor()
    rec = Recommender(rp)
    rec.fit(uim)
    #mse, mae, prec, rec, f = rec.evaluate(uim_test, 20)
    return rec.evaluate(uim_test, 20)

def crossValid(uim1, uim_test1, fold=10):


    uim = uim1
    uim_test = uim_test1

    sliced = uim.data_frame.shape[0]//fold
    x = np.arange(0, uim.data_frame.shape[0], sliced)
    #[0 korak 2*korak .... ]
    #izberes prvega, 0 do korak
    #drugi je korak do 2korak
    #...

    df = uim.data_frame.copy()

    mse, mae, prec, rec, f = 0, 0, 0, 0, 0
    for i in range(len(x)-1):

        if i > 1 and i < len(x)-2:
            uim_t = df.iloc[x[i]:x[i + 1], :]
            uim_learn = df.iloc[x[0]:x[i], :].append(uim.data_frame.iloc[x[i+1]:, :])
        elif i == 0:
            uim_t = uim.data_frame.iloc[x[i]:x[i + 1], :]
            uim_learn = uim.data_frame.iloc[x[i+1]:, :]
        else:
            uim_t = uim.data_frame.iloc[x[i]:, :]
            uim_learn = uim.data_frame.iloc[:x[i], :]

        #klici predikt in pristevaj
        uim.data_frame = uim_learn
        uim_test.data_frame = uim_t
        print(i)
        m, me, pr, rc, fe = predikt(uim, uim_test)
        print("done ...")
        mse+=m
        mae+=me
        prec += pr
        rec += rc
        f += fe
    return mse/fold, mae/fold, prec/fold, rec/fold, f/fold


def incremental(uim1, uim_test1):
    import datetime

    uim = uim1
    uim_test = uim_test1
    df = uim.data_frame.copy()

    #selekcioniras tako kot zgoraj, le da zdaj po datumu, zacnes pri x datumu, vse pred x uporabis za ucenje, po x pa do 2 mesecev pa za test. Potem te dva pridruzis in
    #ponovis ucenje na povecani mnozici, test pa na naslednjih dveh mesecih

    df = uim1.data_frame

    start_date = datetime.date(2002, 12, 10)
    max_date = max(df["dated"])
    end_test = start_date #dummy
    delta = datetime.timedelta(days=5*7)
    mse, mae, prec, rec, f = 0, 0, 0, 0, 0
    count = 0
    while(end_test < max_date):

        start_test = start_date + datetime.timedelta(days=1)
        end_test = start_date + delta
        if end_test >= max_date:
            end_test = max_date

        #koda
        uim_t = df.loc[np.logical_and(df["dated"] >= start_test, df["dated"] <= end_test), :]
        uim_learn = df.loc[df["dated"] <= start_date, :]

        uim.data_frame = uim_learn
        uim_test.data_frame = uim_t
        m, me, pr, rc, fe = predikt(uim, uim_test)
        print(end_test, max_date,"done ...")
        mse+=m
        mae+=me
        prec += pr
        rec += rc
        f += fe
        start_date = end_test+datetime.timedelta(days=1)
        count+=1
    return mse/count, mae/count, prec/count, rec/count, f/count



#uim = UserItemData('data/user_ratedmovies.dat', min_ratings=1000)
#uim.save_data()
uim = UserItemData('saved_user')
uim_test = UserItemData('saved_user')
#md = MovieData('data/movies.dat')
print("done loading users")
#print(crossValid(uim, uim_test))
#print(incremental(uim, uim_test))


# incremental (0.60299358421087101, 0.55733532763832327, 0.1369712611280809, 0.42425073575235656, 0.2060301426127122)
# cross (0.32813831273369343, 0.40532686894955977, 0.29603145000029607, 0.20709944727843294, 0.17589325404800815)








