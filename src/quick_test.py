from .inference import predict

examples = [
    ["itching", "skin rash"],
    ["stomach pain", "acidity", "vomiting"],
]

for ex in examples:
    print(ex, '->', predict(ex))
