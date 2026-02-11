#include <stdio.h>
#include <stdbool.h>
#include <string.h>

bool check_a(const char* str) {
    return (strlen(str) == 1 && str[0] == 'a');
}

bool check_abb(const char* str) {
    return (strcmp(str, "abb") == 0);
}
bool check_a_star_b_plus(const char* str) {
    int len = strlen(str);
    if (len == 0) return false;

    int i = 0;
    // Consume 'a's
    while (i < len && str[i] == 'a') {
        i++;
    }

    // Consume 'b's
    while (i < len && str[i] == 'b') {
        i++;
    }

    // If we've reached the end of the string and consumed at least one 'b', it's a match
    return (i == len && strchr(str, 'b') != NULL);
}

// Main validation function that combines all checks
bool is_valid(const char* str) {
    return check_a(str) || check_abb(str) || check_a_star_b_plus(str);
}

int main() {
    // Test cases
    const char* test_strings[] = {"a", "abb", "b", "ab", "aab", "abbb", "aaabbb", "c", "ababa", "aabbc", ""};
    int num_tests = sizeof(test_strings) / sizeof(test_strings[0]);

    printf("String Recognition Test Results:\n");
    for (int i = 0; i < num_tests; i++) {
        printf("String \"%s\": %s\n", test_strings[i], is_valid(test_strings[i]) ? "Recognized" : "Not Recognized");
    }

    return 0;
}
