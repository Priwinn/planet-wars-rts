import statistics
def get_lesson_length(file_path):
    lessons = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Lesson completed"):
                value = line.split(' ')[3]
                lesson_type = line.split(' ')[5:]
                lesson_type = get_lesson_type(' '.join(lesson_type))
                lessons.append((lesson_type, int(value)))
    return lessons

def get_lesson_type(string):
    if "'greedy'" in string:
        return 'random'
    if "'careful_random'" in string:
        return 'greedy'
    if "'better_greedy'" in string:
        return 'careful_random'
    if "'galactic'" in string:
        return 'better_greedy'
    if "'baseline_buffer'" in string:
        return 'galactic'
    if "updating" in string:
        return 'baseline_buffer'
    
def get_stats(lessons):
    for lesson in lessons[:5]:
        print(f"Lesson type: {lesson[0]}, Length: {lesson[1]}")

    print("Total lesson lengths before league:", sum([lesson[1] for lesson in lessons[:5]]))
    selfplay_lengths = [lesson[1] for lesson in lessons if lesson[0] == 'baseline_buffer']
    if selfplay_lengths:
        avg_length = sum(selfplay_lengths) / len(selfplay_lengths)
        print(f"Selfplay lessons - Avg length: {avg_length}")
        print(f"Selfplay lessons - STD length: {statistics.stdev(selfplay_lengths)}")
    print("Selfplay number of updates:", len(selfplay_lengths))

if __name__ == "__main__":
    file_path = 'logs/output (1).log'
    lessons = get_lesson_length(file_path)
    get_stats(lessons)
    file_path = 'logs/output (2).log'
    lessons = get_lesson_length(file_path)
    get_stats(lessons)
