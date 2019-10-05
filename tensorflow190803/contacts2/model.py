from contacts2.contact import Contact2
class Contacts2Model:
    def __init__(self):
        pass

    @staticmethod
    def set_info(to_do, to_eat):
        contact = Contact2(to_do, to_eat)
        return contact

    @staticmethod
    def get_info(params):
        contacts = []
        for i in params:
            contacts.append(i.to_string())
        return ''.join(contacts)