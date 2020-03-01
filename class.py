#%%
import datetime

#%%
class Employee:

    raise_amount = 1.04
    num_of_emps = 0

    def __init__(self, first, last, pay):
        self.first = first  # These are instance variables
        self.last = last
        self.pay = pay
        self.email = first + last + "." + "@comapny.com"
        Employee.num_of_emps += 1

    def fullname(self): # method for printing out the full name
        return(f"{self.first} {self.last}")

    def apply_raise(self):
        self.pay = int(self.pay * self.raise_amount)

    @classmethod # Creating a class method
    def set_raise_amt(cls, amount):
        cls.raise_amt = amount

    @classmethod # class method as alternative constructor
    def from_string(cls, emp_str):
        first, last, pay = emp_str.split("-")
        return cls(first, last, pay)

    @staticmethod # method that doesn't operate on an instance or an object
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6:
            return False
        else:
            return True

    def __repr__(self):
        return f"Employee({self.first}, {self.last}, {self.pay})"

    def __str__(self):
        return f"{self.fullname()}, {self.email}"

class Developer(Employee):
    raise_amt = 1.10

    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay) # letting Employee class to handle first, last, and pay attributes
        self.prog_lang = prog_lang

class Manager(Employee):

    def __init__(self, first, last, pay, employees = None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)

    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
        print("-->", emp.fullname())

#%%
emp_1 = Employee("Corey", "Schafer", 50000)
print(emp_1)

#%%
mgr_1 = Manager("Alice", "Zhu", 90000)
mgr_1.add_emp(dev_1)
mgr_1.print_emps()

isinstance(mgr_1, Manager)
isinstance(mgr_1, Developer)
issubclass(Manager, Employee)

dev_1 = Developer("David", "Ku", 120000, "python")
dev_2 = Developer("Dau", "Ku", 150000, "java")
print(dev_1.prog_lang)
print(dev_1.email)

print(dev_1.pay)
dev_1.apply_raise()
print(dev_1.pay)

emp_1 = Employee("David", "Ku", 120000)
emp_2 = Employee("Dau", "Ku", 150000)

emp_str_1 = "John-Doe-70000"
emp_str_2 = "Steve-Smith-30000"
emp_str_3 = "Jane-Doe-90000"

emp_3 = Employee.from_string(emp_str_3)
print(emp_3.first)
# emp_1.raise_amount = 1.04 # this create an attribute for employee 1

# print(emp_1.__dict__)

# print(Employee.raise_amount)
# print(emp_1.raise_amount)

my_date = datetime.date(2020, 1, 27)
print(Employee.is_workday(my_date))

a = np.array([5, 5, 5, 5, 5])
for i, j in enumerate(a):
    print((i, j))

#%%
vars = ind_sorted[ind_sorted["clusters_x"] == 3]["var"]
corr = Corr.loc[vars, vars]
corr = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool)).stack().reset_index()
cluster_num = np.ones(corr.shape[0])*3
cluster_num = cluster_num.reshape(corr.shape[0], -1)
corr = np.concatenate([corr, cluster_num], axis = 1)
corr = pd.DataFrame(corr)
corr.columns = ["row", "columns", "corr", "clusters"]
corr = corr[corr["corr"] != 1]

corr_df = corr.copy()

corr_df = pd.concat([corr_df, corr])

#%%
empty = np.empty(1)