from tkinter import *
from ShowMeWhatYouGot import ShowMeWhatYouGot

class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)               
        self.master = master
        self.fields = "question", "answer0", "answer1", "answer2", "answer3", "support"
        self.default_values = "Meiosis is part of the process of gametogenesis, which is the production of what?", "sperm and egg", "chromosomes", "egg only", "sperm only", "Gametogenesis (Spermatogenesis and Oogenesis) Gametogenesis, the production of sperm and eggs, involves the process of meiosis. During meiosis, two nuclear divisions separate the paired chromosomes in the nucleus and then separate the chromatids that were made during an earlier stage of the cell\u2019s life cycle. Meiosis and its associated cell divisions produces haploid cells with half of each pair of chromosomes normally found in diploid cells. The production of sperm is called spermatogenesis and the production of eggs is called oogenesis. Spermatogenesis Spermatogenesis occurs in the wall of the seminiferous tubules, with the most primitive cells at the periphery of the tube and the most mature sperm at the lumen of the tube (Figure 18.14). Immediately under the capsule of the tubule are diploid, undifferentiated cells. These stem cells, each called a spermatogonium (pl. spermatogonia), go through mitosis to produce one cell that remains as a stem cell and a second cell called a primary spermatocyte that will undergo meiosis to produce sperm. The diploid primary spermatocyte goes through meiosis I to produce two haploid cells called secondary spermatocytes. Each secondary spermatocyte divides after meiosis II to produce two cells called spermatids. The spermatids eventually reach the lumen of the tubule and grow a flagellum, becoming sperm cells. Four sperm result from each primary spermatocyte that goes through meiosis."
        self.ents = self.makeform(root)
        self.text = self.make_buttons_and_text()
        self.smwyg = ShowMeWhatYouGot(model_weights='sciq_w.h5', vectors='GoogleNews-vectors-negative300.bin.gz')
            
    def fetch(self, entries):
        dictio = {}
        for entry in entries:
            field = entry[0]
            text  = entry[1].get()
            dictio[field] = text
        odp, prop = self.smwyg.get_answer(dictio)
        self.text.insert(END, "Answer: " + str(odp) + "\n")
        self.text.insert(END, "Probability: " + str(prop) + "\n")

    def makeform(self, root):
        entries = []
        for field, val in zip(self.fields, self.default_values):
            row = Frame(root)
            lab = Label(row, width=15, text=field, anchor='w')
            ent = Entry(row)
            ent.insert(0, val)
            row.pack(side=TOP, fill=X, padx=5, pady=5)
            lab.pack(side=LEFT)
            ent.pack(side=RIGHT, expand=YES, fill=X)
            entries.append((field, ent))
        return entries

    def make_buttons_and_text(self):
        self.master.bind('<Return>', (lambda event, e=self.ents: self.fetch(e)))   
        
        b1 = Button(self.master, text='Answer',
        command=(lambda e=self.ents: self.fetch(e)))
        b1.pack(side=LEFT, padx=5, pady=5)
        
        b2 = Button(self.master, text='Quit', command=self.master.quit)
        b2.pack(side=LEFT, padx=5, pady=5)

        text = Text(root)
        text.pack()

        return text

        # b3 = Button(self.master, text='Get the answers', command=self.get_the_answers)
        # b3.pack(side=LEFT, padx=5, pady=5)


if __name__ == '__main__':
    root = Tk()
    root.geometry("600x600")

    app = Window(root)
    root.mainloop()