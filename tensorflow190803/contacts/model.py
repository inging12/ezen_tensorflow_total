from contacts.contact import Contact
class ContactsModel:
    def __init__(self):
        pass

    #self 안쓰면 self 지워버리고 @staticmethod 선언
    @staticmethod
    def set_contact(name, phone, email, addr):
       contact = Contact(name, phone, email, addr)
       return contact

    @staticmethod
    def get_contacts(params):
        contacts = []
        for i in params:
            contacts.append(i.to_string())
        return ''.join(contacts)

    @staticmethod
    def del_contact(params, name):
        for i, t in enumerate(params):
            if t.name == name:
                del params[i]
