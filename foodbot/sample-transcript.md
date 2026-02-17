# Sample Transcript — FoodBot Demo

> This transcript demonstrates structured input, free-text AI parsing, ambiguity handling, and full checkout flow.

---

## 1. Entry & Help

```
User: /start

Bot:  👋 Welcome to FoodBot, Ahmed!

      I'm your personal food ordering assistant. Browse our menu, add
      items to your cart, and checkout when ready.

      Use the buttons below or type items directly like:
      • BG1 x2 — add 2 Classic Beef Burgers
      • "I want a chicken burger and fries" — I'll figure it out!

      What would you like to do?

      [📋 View Menu] [🛒 My Cart]
      [✅ Checkout]   [❓ Help]
```

```
User: [taps ❓ Help]

Bot:  ❓ FoodBot Help

      Commands:
      /start — Main menu
      /menu — View our menu
      /cart — View your cart
      /checkout — Proceed to checkout
      /clear — Clear your cart
      /help — This help message

      Ordering:
      • Tap menu buttons to add items
      • Type item codes: BG1 x2 or SD1, DR2
      • Or describe what you want: "I'd like a chicken burger
        with fries and a coke"

      Prices are in Egyptian Pounds (EGP)

      [📋 View Menu] [🛒 My Cart]
```

---

## 2. Menu Browsing

```
User: [taps 📋 View Menu]

Bot:  📋 Our Menu

      Choose a category to browse:

      [🍔 Burgers]
      [🍟 Sides]
      [🥤 Drinks]
      [🛒 My Cart]
```

```
User: [taps 🍔 Burgers]

Bot:  Burgers 🍔

      BG1 — Classic Beef Burger
      Angus beef, lettuce, tomato, pickles
      💰 85 EGP

      BG2 — Chicken Burger
      Crispy chicken, coleslaw, mayo
      💰 75 EGP

      BG3 — Double Smash Burger
      Two patties, cheese, special sauce
      💰 120 EGP

      BG4 — Veggie Burger
      Halloumi, peppers, hummus
      💰 70 EGP

      Tap a button to add, or type e.g. BG1 x2

      [➕ BG1 - Classic Beef Burger (85 EGP)]
      [➕ BG2 - Chicken Burger (75 EGP)]
      [➕ BG3 - Double Smash Burger (120 EGP)]
      [➕ BG4 - Veggie Burger (70 EGP)]
      [⬅️ Back to Menu] [🛒 My Cart]
```

---

## 3. Add to Cart — Button

```
User: [taps ➕ BG1 - Classic Beef Burger (85 EGP)]

Bot:  ✅ Added to cart:
      • 1x Classic Beef Burger (85 EGP)

      🛒 Cart total: 85 EGP (1 items)

      [📋 View Menu] [🛒 My Cart]
      [✅ Checkout]
```

---

## 4. Add to Cart — Structured Codes

```
User: BG3 x2, DR1

Bot:  ✅ Added to cart:
      • 2x Double Smash Burger (240 EGP)
      • 1x Coca-Cola (25 EGP)

      📝 Parsed from code
      🛒 Cart total: 350 EGP (3 items)

      [📋 View Menu] [🛒 My Cart]
      [✅ Checkout]
```

---

## 5. Add to Cart — Free-Text (AI Extraction)

```
User: I'd like some fries and a mango smoothie please

Bot:  ✅ Added to cart:
      • 1x French Fries (30 EGP)
      • 1x Mango Smoothie (45 EGP)

      🤖 Parsed with AI
      🛒 Cart total: 425 EGP (5 items)

      [📋 View Menu] [🛒 My Cart]
      [✅ Checkout]
```

---

## 6. Ambiguity Handling (AI)

```
User: give me a burger

Bot:  ✅ Added to cart:
      • 1x Classic Beef Burger (85 EGP)

      🤖 Parsed with AI
      🛒 Cart total: 510 EGP (6 items)

      [📋 View Menu] [🛒 My Cart]
      [✅ Checkout]
```

> Note: The AI picks the most likely match (Classic Beef Burger) for the
> ambiguous term "a burger". This is by design — the system prompt tells
> the model to pick the best match.

---

## 7. Unrecognized Input

```
User: do you have pizza?

Bot:  🤔 I couldn't find any menu items in your message.

      Try:
      • Item codes like BG1 x2
      • Or descriptions like "chicken burger and fries"

      Type /menu to see what's available!

      [📋 View Menu] [❓ Help]
```

---

## 8. View Cart

```
User: [taps 🛒 My Cart]

Bot:  🛒 Your Cart

      1. Classic Beef Burger (BG1)
         2x 85 EGP = 170 EGP
      2. Double Smash Burger (BG3)
         2x 120 EGP = 240 EGP
      3. Coca-Cola (DR1)
         1x 25 EGP = 25 EGP
      4. French Fries (SD1)
         1x 30 EGP = 30 EGP
      5. Mango Smoothie (DR3)
         1x 45 EGP = 45 EGP

      💰 Total: 510 EGP

      [📋 Add More] [🗑 Clear Cart]
      [✅ Checkout]
```

---

## 9. Checkout Flow

```
User: [taps ✅ Checkout]

Bot:  🧾 Checkout

      • 2x Classic Beef Burger — 170 EGP
      • 2x Double Smash Burger — 240 EGP
      • 1x Coca-Cola — 25 EGP
      • 1x French Fries — 30 EGP
      • 1x Mango Smoothie — 45 EGP

      💰 Total: 510 EGP

      How would you like to receive your order?

      [🏪 Pickup] [🚗 Delivery]
      [❌ Cancel]
```

```
User: [taps 🚗 Delivery]

Bot:  🚗 Got it — delivery!

      Please type your full name:
```

```
User: Ahmed Hassan

Bot:  👤 Name: Ahmed Hassan

      Now please enter your phone number
      (Egyptian format, e.g. 01012345678):
```

```
User: 0123456

Bot:  ❌ Invalid phone number. Please enter a valid Egyptian mobile
      number.
      Example: 01012345678 or +201012345678
```

```
User: 01012345678

Bot:  📱 Phone: 01012345678

      Please enter your delivery address:
```

```
User: 15 El-Tahrir St, Dokki, Giza

Bot:  📋 Order Summary

      • 2x Classic Beef Burger — 170 EGP
      • 2x Double Smash Burger — 240 EGP
      • 1x Coca-Cola — 25 EGP
      • 1x French Fries — 30 EGP
      • 1x Mango Smoothie — 45 EGP

      💰 Total: 510 EGP

      🚗 Delivery
      👤 Ahmed Hassan
      📱 01012345678
      📍 15 El-Tahrir St, Dokki, Giza

      Confirm this order?

      [✅ Confirm Order] [❌ Cancel]
```

```
User: [taps ✅ Confirm Order]

Bot:  🎉 Order Confirmed!

      🆔 Order ID: ORD-1001

      • 2x Classic Beef Burger — 170 EGP
      • 2x Double Smash Burger — 240 EGP
      • 1x Coca-Cola — 25 EGP
      • 1x French Fries — 30 EGP
      • 1x Mango Smoothie — 45 EGP

      💰 Total: 510 EGP

      🚗 Delivery
      👤 Ahmed Hassan
      📱 01012345678
      📍 15 El-Tahrir St, Dokki, Giza

      ⏱ Estimated time: 30-45 minutes

      Thank you for your order! 🙏

      [📋 New Order]
```

---

## 10. Empty Cart Edge Case

```
User: /checkout

Bot:  🛒 Your cart is empty! Add some items first.

      [📋 View Menu]
```

---

## 11. AI Fallback (When Ollama is Down)

```
User: can I get a chicken burger and some onion rings

Bot:  ✅ Added to cart:
      • 1x Chicken Burger (75 EGP)
      • 1x Onion Rings (35 EGP)

      🔍 Matched by keywords
      🛒 Cart total: 110 EGP (2 items)

      [📋 View Menu] [🛒 My Cart]
      [✅ Checkout]
```

> Note: When the AI is unavailable, the keyword fallback parser
> successfully matches "chicken burger" → BG2 and "onion rings" → SD2.
