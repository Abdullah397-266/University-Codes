#include <iostream>
#include <cstring>
#include <ctime>
#include <windows.h>
#include <conio.h>
using namespace std;

//consts for array sizes and limits
const int MAX_EXPENSES = 1000;
const int MAX_CATEGORIES = 10;
const int MAX_NAME_LENGTH = 50;
const int MAX_DESCRIPTION_LENGTH = 100;

// Expense Category Structure
struct ExpenseCategory
{
    char name[MAX_NAME_LENGTH];
    double budgetLimit;
    double totalSpent;
};

// Expense Entry Structure
struct ExpenseEntry
{
    char description[MAX_DESCRIPTION_LENGTH];
    double amount;
    char category[MAX_NAME_LENGTH];
    time_t timestamp;
};

// Student Expense Tracker Class
class StudentExpenseTracker
{
private:
    ExpenseCategory categories[MAX_CATEGORIES];
    ExpenseEntry expenses[MAX_EXPENSES];
    int categoryCount;
    int expenseCount;
    double totalIncome;

    // Private helper methods
    int findCategoryIndex(char *categoryName)
    {
        for (int i = 0; i < categoryCount; ++i)
        {
            if (strcmp(categories[i].name, categoryName) == 0)
            {
                return i;
            }
        }
        return -1;
    }

    bool isCategoryExists(char *categoryName)
    {
        return findCategoryIndex(categoryName) != -1;
    }

public:
    // ructor
    StudentExpenseTracker()
    {
        categoryCount = 0;
        expenseCount = 0;
        totalIncome = 0.0;
    }

    // Method to add a new expense category
    bool addExpenseCategory(char *name, double budgetLimit)
    {
        if (categoryCount >= MAX_CATEGORIES)
        {
            cout << "Maximum number of categories reached!" << endl;
            return false;
        }

        if (isCategoryExists(name))
        {
            cout << "Category already exists!" << endl;
            return false;
        }

        strncpy(categories[categoryCount].name, name, MAX_NAME_LENGTH - 1);
        categories[categoryCount].name[MAX_NAME_LENGTH - 1] = '\0'; // Ensure null-termination
        categories[categoryCount].budgetLimit = budgetLimit;
        categories[categoryCount].totalSpent = 0.0;
        categoryCount++;
        return true;
    }

    // Method to add income
    void addIncome(double amount)
    {
        totalIncome += amount;
        cout << "Income added: $" << amount << endl;
    }

    // Method to record an expense
    bool recordExpense(char *description, double amount, char *category)
    {
        if (expenseCount >= MAX_EXPENSES)
        {
            cout << "Maximum number of expenses reached!" << endl;
            return false;
        }

        int categoryIndex = findCategoryIndex(category);
        if (categoryIndex == -1)
        {
            cout << "Category does not exist!" << endl;
            return false;
        }

        // Check budget limit
        if (categories[categoryIndex].totalSpent + amount > categories[categoryIndex].budgetLimit)
        {
            cout << "Warning: Expense exceeds category budget limit!" << endl;
        }

        // Record expense
        strncpy(expenses[expenseCount].description, description, MAX_DESCRIPTION_LENGTH - 1);
        expenses[expenseCount].description[MAX_DESCRIPTION_LENGTH - 1] = '\0'; // Ensure null-termination
        expenses[expenseCount].amount = amount;

        strncpy(expenses[expenseCount].category, category, MAX_NAME_LENGTH - 1);
        expenses[expenseCount].category[MAX_NAME_LENGTH - 1] = '\0'; // Ensure null-termination

        expenses[expenseCount].timestamp = time(nullptr);

        // Update category total spent
        categories[categoryIndex].totalSpent += amount;
        expenseCount++;

        cout << "Expense recorded successfully!" << endl;
        return true;
    }

    // Method to display expense summary
    void displayExpenseSummary()
    {
        cout << "\n--- Expense Summary ---" << endl;
        cout << "Total Income: $" << totalIncome << endl;

        double totalExpenses = 0.0;
        for (int i = 0; i < categoryCount; ++i)
        {
            cout << "\nCategory: " << categories[i].name << endl;
            cout << "Budget Limit: $" << categories[i].budgetLimit << endl;
            cout << "Total Spent: $" << categories[i].totalSpent << endl;

            if (categories[i].totalSpent > categories[i].budgetLimit)
            {
                cout << "BUDGET EXCEEDED!" << endl;
            }

            totalExpenses += categories[i].totalSpent;
        }

        cout << "\nTotal Expenses: $" << totalExpenses << endl;
        cout << "Remaining Budget: $" << (totalIncome - totalExpenses) << endl;
    }

    // Method to display expense history
    void displayExpenseHistory()
    {
        cout << "\n--- Expense History ---" << endl;
        for (int i = 0; i < expenseCount; ++i)
        {
            char timeStr[26];
            ctime_s(timeStr, sizeof(timeStr), &expenses[i].timestamp);

            cout << "Description: " << expenses[i].description << endl;
            cout << "Amount: $" << expenses[i].amount << endl;
            cout << "Category: " << expenses[i].category << endl;
            cout << "Date: " << timeStr;
            cout << "------------------------" << endl;
        }
    }
};

// Main application with interactive menu
void displayMenu()
{
    cout << "\n--- Student Expense Tracker ---" << endl;
    cout << "1. Add Expense Category" << endl;
    cout << "2. Add Income" << endl;
    cout << "3. Record Expense" << endl;
    cout << "4. Display Expense Summary" << endl;
    cout << "5. Display Expense History" << endl;
    cout << "6. Exit" << endl;
    cout << "Enter your choice: ";
}

int main()
{
    StudentExpenseTracker tracker;
    int choice;
    char name[MAX_NAME_LENGTH];
    char description[MAX_DESCRIPTION_LENGTH];
    double amount, budgetLimit;

    while (true)
    {
        system("CLS"); // Clear screen

        displayMenu();
        cin >> choice;
        cin.ignore(); // Clear input buffer

        switch (choice)
        {
        case 1:
            cout << "\nEnter category name: ";
            cin.getline(name, MAX_NAME_LENGTH);

            cout << "Enter budget limit: ";
            cin >> budgetLimit;
            cin.ignore(); // Clear input buffer

            tracker.addExpenseCategory(name, budgetLimit);
            break;

        case 2:
            cout << "\nEnter income amount: ";
            cin >> amount;
            cin.ignore(); // Clear input buffer

            tracker.addIncome(amount);
            break;

        case 3:
            cout << "\nEnter category: ";
            cin.getline(name, MAX_NAME_LENGTH);

            cout << "Enter expense amount: ";
            cin >> amount;
            cin.ignore(); // Clear input buffer

            cout << "Enter description: ";
            cin.getline(description, MAX_DESCRIPTION_LENGTH);

            tracker.recordExpense(description, amount, name);
            cout << "------------------------" << endl;
            cout << "Press any key to continue...";
            _getch(); // Wait for key press (halting the program)
            break;

        case 4:
            tracker.displayExpenseSummary();
            cout << "------------------------" << endl;
            cout << "Press any key to continue...";
            _getch(); // Wait for key press (halting the program)
            break;

        case 5:
            tracker.displayExpenseHistory();
            cout << "Press any key to continue...";
            _getch(); // Wait for key press (halting the program)
            break;

        case 6:
            cout << "\nExiting Expense Tracker. Goodbye!" << endl;
            return 0;

        default:
            cout << "\nInvalid choice. Please try again." << endl;
        }
    }

    return 0;
}