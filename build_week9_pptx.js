const pptxgen = require("pptxgenjs");
const BG="0D1229",CARD_BG="1E2341",BLUE="60A5FA",PURPLE="A78BFA",GREEN="4ADE80",ORANGE="FBBF24",CYAN="22D3EE",PINK="F472B6",RED="F87171",GOLD="FBBF24",GRAY="8B95B0",LIGHT="B0B8D0",WHITE="FFFFFF";
function addTagline(s,t,y){y=y||4.95;s.addShape("rect",{x:0.4,y,w:9.2,h:0.4,fill:{color:"141E32"},line:{color:"2A3560",width:0.5},rectRadius:0.05});s.addText(t,{x:0.4,y,w:9.2,h:0.4,fontSize:11,fontFace:"Arial",bold:true,color:GOLD,align:"center",valign:"middle"});}
function addCard(s,title,body,x,y,w,h,ac){s.addShape("rect",{x,y,w,h,fill:{color:CARD_BG},rectRadius:0.08});s.addShape("rect",{x,y,w:0.06,h,fill:{color:ac||BLUE}});s.addText(title,{x:x+0.15,y,w:w-0.2,h:0.35,fontSize:13,fontFace:"Arial",bold:true,color:ac||BLUE,valign:"top",margin:[4,0,0,0]});s.addText(body,{x:x+0.15,y:y+0.32,w:w-0.2,h:h-0.36,fontSize:10,fontFace:"Arial",color:LIGHT,valign:"top",lineSpacingMultiple:1.3});}
function addNumberedItem(s,n,title,desc,x,y,w,cc,tc){s.addShape("rect",{x,y,w,h:0.55,fill:{color:CARD_BG},rectRadius:0.06});s.addShape("oval",{x:x+0.1,y:y+0.1,w:0.35,h:0.35,fill:{color:cc}});s.addText(String(n),{x:x+0.1,y:y+0.1,w:0.35,h:0.35,fontSize:11,fontFace:"Arial",bold:true,color:WHITE,align:"center",valign:"middle"});s.addText(title,{x:x+0.55,y:y+0.05,w:w-0.65,h:0.22,fontSize:11,fontFace:"Arial",bold:true,color:tc||BLUE});s.addText(desc,{x:x+0.55,y:y+0.27,w:w-0.65,h:0.23,fontSize:9,fontFace:"Arial",color:LIGHT});}
function titleSlide(p,t,sub,wk){let s=p.addSlide();s.background={color:BG};s.addShape("oval",{x:2.5,y:-0.5,w:3,h:3,fill:{color:"8B5CF6",transparency:88}});s.addShape("oval",{x:6,y:3.5,w:2.5,h:2.5,fill:{color:"3B82F6",transparency:88}});s.addText("DEVOPS WITH VIBECODING",{x:0,y:1.2,w:10,h:0.35,fontSize:11,fontFace:"Arial",color:"8B5CF6",align:"center",charSpacing:4});s.addText(t,{x:0.5,y:1.7,w:9,h:0.6,fontSize:28,fontFace:"Arial",bold:true,color:WHITE,align:"center"});s.addText(sub,{x:1,y:2.4,w:8,h:0.4,fontSize:16,fontFace:"Arial",color:BLUE,align:"center"});s.addText(wk,{x:3.5,y:3.1,w:3,h:0.35,fontSize:12,fontFace:"Arial",color:GRAY,align:"center"});s.addText("Anirach Mingkhwan",{x:2,y:3.6,w:6,h:0.3,fontSize:11,fontFace:"Arial",color:GRAY,align:"center"});s.addText("FITM, KMUTNB",{x:2,y:3.9,w:6,h:0.3,fontSize:10,fontFace:"Arial",color:GRAY,align:"center"});}
function sectionSlide(p,num,t,sub){let s=p.addSlide();s.background={color:BG};s.addShape("rect",{x:0,y:0,w:10,h:5.63,fill:{color:"111936"}});s.addShape("oval",{x:-1,y:1,w:4,h:4,fill:{color:"8B5CF6",transparency:92}});s.addShape("oval",{x:7,y:-0.5,w:3,h:3,fill:{color:"3B82F6",transparency:92}});s.addText(num,{x:3.5,y:1.5,w:3,h:0.5,fontSize:14,fontFace:"Arial",color:PURPLE,align:"center",charSpacing:3});s.addText(t,{x:1,y:2.1,w:8,h:0.6,fontSize:28,fontFace:"Arial",bold:true,color:WHITE,align:"center"});s.addText(sub||"",{x:1.5,y:2.8,w:7,h:0.4,fontSize:14,fontFace:"Arial",color:BLUE,align:"center"});}
function contentSlide(p,t){let s=p.addSlide();s.background={color:BG};s.addShape("rect",{x:0,y:0,w:10,h:0.9,fill:{color:"111936"}});s.addShape("rect",{x:0,y:0.88,w:10,h:0.03,fill:{color:PURPLE,transparency:50}});s.addText(t,{x:0.5,y:0.15,w:9,h:0.6,fontSize:20,fontFace:"Arial",bold:true,color:WHITE});return s;}

let pres = new pptxgen();
pres.layout = "LAYOUT_16x9";

// 1: Title
titleSlide(pres, "Automated Testing\nwith AI", "AI-Augmented Quality Assurance", "Week 9");

// 2: Learning Objectives
let s = contentSlide(pres, "Learning Objectives");
addNumberedItem(s, 1, "Testing Strategies", "Design unit, integration, and E2E testing approaches", 0.5, 1.1, 9, PURPLE, PURPLE);
addNumberedItem(s, 2, "AI Test Generation", "Use AI for comprehensive test case creation with edge cases", 0.5, 1.75, 9, BLUE, BLUE);
addNumberedItem(s, 3, "CI/CD Test Automation", "Implement coverage enforcement and parallel execution", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "Mutation Testing", "Validate test suite quality with mutation analysis", 0.5, 3.05, 9, ORANGE, ORANGE);
addNumberedItem(s, 5, "Critical Evaluation", "Assess and improve AI-generated tests", 0.5, 3.7, 9, RED, RED);
addTagline(s, '"Tests are the safety net that lets you move fast with confidence"');

// 3: Agenda
s = contentSlide(pres, "Today's Agenda");
addCard(s, "Part 1: Foundations", "Testing Pyramid\nUnit / Integration / E2E\nContract Testing", 0.5, 1.1, 4.3, 1.5, PURPLE);
addCard(s, "Part 2: AI Testing", "AI-Generated Test Cases\nEdge Cases & Properties\nCritical Review Process", 5.2, 1.1, 4.3, 1.5, BLUE);
addCard(s, "Part 3: CI/CD", "Coverage Enforcement\nParallel Execution\nTest Automation Pipeline", 0.5, 2.8, 4.3, 1.5, GREEN);
addCard(s, "Part 4: Advanced", "Mutation Testing\nProperty-Based Testing\nTDD with AI + Lab", 5.2, 2.8, 4.3, 1.5, ORANGE);

// === SECTION 1: TESTING PYRAMID ===
sectionSlide(pres, "SECTION 01", "The Testing Pyramid", "Foundation of Quality Assurance");

// 5: Testing Pyramid
s = contentSlide(pres, "The Testing Pyramid");
addCard(s, "Unit Tests (Base - Many)", "- Fast, deterministic, independent\n- Test single functions/methods\n- Mock external dependencies\n- Run in milliseconds\n- Framework: pytest, Jest, JUnit\n- Target: 70-80% of all tests", 0.5, 1.1, 2.8, 2.8, GREEN);
addCard(s, "Integration Tests (Middle)", "- Test component interactions\n- Database, API, message queues\n- Slower, need infrastructure\n- Docker Compose for dependencies\n- Framework: pytest + requests\n- Target: 15-20% of all tests", 3.6, 1.1, 2.8, 2.8, BLUE);
addCard(s, "E2E Tests (Top - Few)", "- Test complete user journeys\n- Real browser, real services\n- Slow, fragile, expensive\n- Critical paths only\n- Framework: Playwright, Cypress\n- Target: 5-10% of all tests", 6.7, 1.1, 2.8, 2.8, PURPLE);

// 6: Unit Testing Best Practices
s = contentSlide(pres, "Unit Testing with pytest");
addCard(s, "Good Unit Test", "def test_calculate_discount():\n    # Arrange\n    product = Product(price=100)\n    \n    # Act\n    result = product.apply_discount(0.2)\n    \n    # Assert\n    assert result == 80.0\n    assert product.discount_applied == True", 0.5, 1.1, 4.3, 2.5, GREEN);
addCard(s, "Key Principles", "- AAA Pattern: Arrange, Act, Assert\n- One assertion per concept\n- Descriptive test names\n- Independent (no shared state)\n- Fast (< 100ms each)\n- Deterministic (no randomness)\n- Use fixtures for setup/teardown\n- Mock external dependencies", 5.2, 1.1, 4.3, 2.5, BLUE);
addTagline(s, "A test that can't fail is worthless - test behavior, not implementation");

// 7: Integration Testing
s = contentSlide(pres, "Integration Testing");
addCard(s, "Database Integration", "import pytest\nfrom sqlalchemy import create_engine\n\n@pytest.fixture\ndef db_session():\n    engine = create_engine('sqlite:///:memory:')\n    Base.metadata.create_all(engine)\n    session = Session(engine)\n    yield session\n    session.rollback()\n\ndef test_create_user(db_session):\n    user = User(name='test')\n    db_session.add(user)\n    db_session.commit()\n    assert db_session.query(User).count() == 1", 0.5, 1.1, 4.3, 3.2, BLUE);
addCard(s, "API Integration", "import requests\n\ndef test_api_create_order():\n    # Real API call\n    resp = requests.post(\n        'http://localhost:8000/orders',\n        json={'item': 'widget', 'qty': 5}\n    )\n    assert resp.status_code == 201\n    data = resp.json()\n    assert data['item'] == 'widget'\n    assert data['total'] > 0\n\n    # Verify side effects\n    stock = requests.get(f'/stock/widget')\n    assert stock.json()['qty'] == 95", 5.2, 1.1, 4.3, 3.2, PURPLE);

// 8: Contract Testing
s = contentSlide(pres, "Contract Testing with Pact");
addCard(s, "The Problem", "Service A depends on Service B's API\nIntegration tests need both running\nChanges in B can break A silently\nE2E tests are slow and flaky", 0.5, 1.1, 4.3, 1.7, RED);
addCard(s, "Contract Testing Solution", "Consumer defines expected interactions\nProvider verifies it can fulfill them\nNo need to run both services together\nFast, reliable, catches breaking changes", 5.2, 1.1, 4.3, 1.7, GREEN);
addCard(s, "How Pact Works", "1. Consumer writes test with expected request/response\n2. Pact generates contract file (JSON)\n3. Provider runs contract against its API\n4. Both sides verified independently - CI catches mismatches", 0.5, 3.0, 9, 1.4, BLUE);
addTagline(s, "Contract tests verify interfaces without all services running");

// 9: E2E Testing
s = contentSlide(pres, "E2E Testing with Playwright");
addCard(s, "When to Use E2E", "- Critical user journeys only\n- Login, checkout, core workflows\n- Smoke tests for deployments\n- NOT for edge cases (use unit tests)\n- Keep count minimal (< 50 total)", 0.5, 1.1, 4.3, 2.0, PURPLE);
addCard(s, "Playwright Example", "from playwright.sync_api import sync_playwright\n\ndef test_user_login():\n    with sync_playwright() as p:\n        browser = p.chromium.launch()\n        page = browser.new_page()\n        page.goto('http://localhost:3000/login')\n        page.fill('#email', 'test@test.com')\n        page.fill('#password', 'secret')\n        page.click('button[type=submit]')\n        assert page.url == '/dashboard'", 5.2, 1.1, 4.3, 2.0, BLUE);
addTagline(s, "E2E tests: few but critical - guard your most important user flows");

// === SECTION 2: AI TEST GENERATION ===
sectionSlide(pres, "SECTION 02", "AI-Generated Test Cases", "Accelerating Test Creation");

// 11: AI Test Generation
s = contentSlide(pres, "AI for Test Generation");
addCard(s, "What AI Generates Well", "- Normal path test cases\n- Edge cases (boundaries, empty, null)\n- Error handling scenarios\n- Property-based test specifications\n- Mock setup boilerplate\n- Parameterized test variants", 0.5, 1.1, 4.3, 2.3, GREEN);
addCard(s, "Effective Prompting", "Include in your prompt:\n- Source code to test\n- Testing framework (pytest, Jest)\n- Fixture requirements\n- Expected behaviors\n- Mock requirements\n- Edge cases to cover\n- Coverage goals", 5.2, 1.1, 4.3, 2.3, BLUE);
addTagline(s, "AI writes the first draft - you refine and validate");

// 12: AI Prompting Example
s = contentSlide(pres, "Prompting for Tests: Example");
addCard(s, "Good Prompt", '"Write pytest tests for this function:\n\ndef calculate_shipping(weight, destination, express=False):\n    if weight <= 0: raise ValueError\n    base = 5.99 if destination == \'domestic\' else 15.99\n    rate = weight * (0.5 if destination == \'domestic\' else 1.2)\n    total = base + rate\n    if express: total *= 1.5\n    return round(total, 2)\n\nInclude: normal cases, edge cases (0, negative,\nlarge weight), all destinations, express/standard,\nand parameterized variants."', 0.5, 1.1, 9, 3.3, BLUE);
addTagline(s, "Specific prompts = specific, useful tests");

// 13: AI Test Pitfalls
s = contentSlide(pres, "AI Test Generation: Pitfalls");
addCard(s, "Common Problems", "- Tests implementation, not behavior\n  (fragile to refactoring)\n- Missing domain-specific edge cases\n- Meaningless assertions\n  (assert result is not None)\n- Over-complex test setups\n- Tautological tests (testing the mock)\n- Ignoring concurrency issues", 0.5, 1.1, 4.3, 2.5, RED);
addCard(s, "Review Checklist", "For every AI-generated test, verify:\n\n1. Does it test behavior, not implementation?\n2. Would it catch a real bug?\n3. Is the assertion meaningful?\n4. Is it independent of other tests?\n5. Is the setup minimal & clear?\n6. Does it cover a unique scenario?\n7. Would it survive a refactor?", 5.2, 1.1, 4.3, 2.5, GREEN);
addTagline(s, "Bad tests are worse than no tests - they give false confidence");

// 14: AI vs Human Tests
s = contentSlide(pres, "AI vs Human Test Quality");
addCard(s, "AI Tests Excel At", "- Boilerplate generation (fast)\n- Boundary value analysis\n- Exhaustive parameter combinations\n- Standard error scenarios\n- Code coverage completeness\n- Consistent style", 0.5, 1.1, 4.3, 2.0, BLUE);
addCard(s, "Human Tests Excel At", "- Business logic validation\n- Real-world user scenarios\n- Security edge cases\n- Performance expectations\n- Concurrency & race conditions\n- Domain expertise encoding", 5.2, 1.1, 4.3, 2.0, PURPLE);
addCard(s, "Best Approach: Hybrid", "AI generates 80% of tests (coverage) -> Human adds 20% (domain knowledge, critical paths, security)\nHuman reviews ALL AI tests before committing", 0.5, 3.3, 9, 1.2, GREEN);

// === SECTION 3: CI/CD TESTING ===
sectionSlide(pres, "SECTION 03", "Test Automation in CI/CD", "Coverage, Speed & Enforcement");

// 16: Coverage Types
s = contentSlide(pres, "Code Coverage Deep Dive");
addCard(s, "Coverage Types", "Line: % of lines executed\nBranch: % of if/else paths taken\nFunction: % of functions called\nPath: % of unique code paths\n\nBranch coverage > line coverage\n(lines can hide untested branches)", 0.5, 1.1, 4.3, 2.3, BLUE);
addCard(s, "Tools & Configuration", "# Python (coverage.py + pytest)\npytest --cov=myapp --cov-report=html \\\n  --cov-branch --cov-fail-under=80\n\n# JavaScript (Istanbul/c8)\nnyc --branches 80 --lines 80 \\\n  --reporter=html mocha\n\n# Java (JaCoCo)\n# Configure in pom.xml/build.gradle", 5.2, 1.1, 4.3, 2.3, GREEN);
addTagline(s, "80% coverage minimum - but coverage alone doesn't mean quality");

// 17: Coverage Enforcement
s = contentSlide(pres, "Coverage Enforcement in CI");
addCard(s, "GitHub Actions", "- name: Run tests with coverage\n  run: pytest --cov=myapp --cov-branch \\\n    --cov-fail-under=80 --cov-report=xml\n\n- name: Upload coverage\n  uses: codecov/codecov-action@v3\n  with:\n    files: coverage.xml\n    fail_ci_if_error: true", 0.5, 1.1, 9, 1.8, BLUE);
addCard(s, "PR Coverage Rules", "- Block merge if coverage drops\n- Require new code to be covered\n- Show coverage diff in PR comments\n- Track trends over time (Codecov, Coveralls)\n- Set per-directory thresholds for critical modules", 0.5, 3.1, 9, 1.4, PURPLE);

// 18: Parallel Execution
s = contentSlide(pres, "Parallel Test Execution");
addCard(s, "pytest-xdist", "# Run tests across 4 CPU cores\npytest -n 4 tests/\n\n# Auto-detect available cores\npytest -n auto tests/\n\nRequirements:\n- Tests must be independent\n- No shared state between tests\n- Use fixtures for isolation", 0.5, 1.1, 4.3, 2.3, GREEN);
addCard(s, "GitHub Actions Matrix", "strategy:\n  matrix:\n    test-group: [unit, integration, e2e]\n    python-version: [3.11, 3.12]\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n    - run: pytest tests/${{ matrix.test-group }}\n\n# All groups run in parallel!", 5.2, 1.1, 4.3, 2.3, BLUE);
addTagline(s, "Parallel tests = faster feedback = faster development");

// 19: Test Pipeline Architecture
s = contentSlide(pres, "Complete Test Pipeline");
addNumberedItem(s, 1, "Pre-commit", "Linting + formatting + fast unit tests (< 30 sec)", 0.5, 1.1, 9, GREEN, GREEN);
addNumberedItem(s, 2, "CI: Unit Tests", "All unit tests in parallel, coverage report (< 2 min)", 0.5, 1.75, 9, BLUE, BLUE);
addNumberedItem(s, 3, "CI: Integration Tests", "Database + API tests with Docker services (< 5 min)", 0.5, 2.4, 9, PURPLE, PURPLE);
addNumberedItem(s, 4, "CI: E2E Tests", "Critical path tests with Playwright (< 10 min)", 0.5, 3.05, 9, ORANGE, ORANGE);
addNumberedItem(s, 5, "Post-merge", "Full regression + mutation testing (nightly)", 0.5, 3.7, 9, RED, RED);

// === SECTION 4: ADVANCED TESTING ===
sectionSlide(pres, "SECTION 04", "Advanced Testing", "Property-Based & Mutation Testing");

// 21: Property-Based Testing
s = contentSlide(pres, "Property-Based Testing with Hypothesis");
addCard(s, "What is It?", "Instead of specific examples,\ndefine PROPERTIES that must always hold\nFramework generates random inputs\nFinds edge cases you'd never think of\n\nExample properties:\n- Sorting: output is ordered\n- Encoding: decode(encode(x)) == x\n- Math: add(a,b) == add(b,a)", 0.5, 1.1, 4.3, 2.8, PURPLE);
addCard(s, "Hypothesis Example", "from hypothesis import given, strategies as st\n\n@given(st.lists(st.integers()))\ndef test_sort_is_idempotent(xs):\n    assert sorted(sorted(xs)) == sorted(xs)\n\n@given(st.text())\ndef test_encode_decode_roundtrip(s):\n    assert decode(encode(s)) == s\n\n@given(st.integers(min_value=1),\n       st.integers(min_value=1))\ndef test_gcd_divides_both(a, b):\n    g = gcd(a, b)\n    assert a % g == 0\n    assert b % g == 0", 5.2, 1.1, 4.3, 2.8, GREEN);

// 22: Property-Based AI Prompting
s = contentSlide(pres, "AI + Property-Based Testing");
addCard(s, "Prompt for Properties", '"Given this function:\n\ndef serialize(obj): ...\ndef deserialize(data): ...\n\nSuggest 5 properties that should\nalways hold, and write Hypothesis\ntests for each."', 0.5, 1.1, 4.3, 2.0, BLUE);
addCard(s, "AI Suggests Properties", "1. Roundtrip: deserialize(serialize(x)) == x\n2. Idempotent: serialize(x) always same output\n3. Type preservation: types survive roundtrip\n4. Size: serialized size > 0 for non-empty\n5. Deterministic: same input = same output\n\nAI excels at finding universal properties!", 5.2, 1.1, 4.3, 2.0, GREEN);
addTagline(s, "Property-based testing finds bugs that example-based testing misses");

// 23: Mutation Testing
s = contentSlide(pres, "Mutation Testing");
addCard(s, "How It Works", "1. Take your passing test suite\n2. Introduce small code mutations:\n   - Change + to -\n   - Change > to >=\n   - Remove return statement\n   - Change True to False\n3. Run tests against each mutant\n4. If tests still pass = SURVIVING mutant\n5. Surviving mutants = test gaps!", 0.5, 1.1, 4.3, 2.8, PURPLE);
addCard(s, "mutmut (Python)", "# Install\npip install mutmut\n\n# Run mutation testing\nmutmut run --paths-to-mutate=myapp/\n\n# View results\nmutmut results\nmutmut show 42  # inspect surviving mutant\n\n# Metrics\nMutation Score = killed / total * 100%\nTarget: > 80% mutation score", 5.2, 1.1, 4.3, 2.8, GREEN);

// 24: Mutation vs Coverage
s = contentSlide(pres, "Mutation Testing vs Code Coverage");
addCard(s, "Code Coverage", "- Measures: lines/branches executed\n- Can be 100% with no assertions!\n- Fast to compute\n- Necessary but not sufficient\n- \"Did the test run the code?\"", 0.5, 1.1, 4.3, 2.0, BLUE);
addCard(s, "Mutation Testing", "- Measures: can tests detect changes?\n- Actually validates test quality\n- Expensive to compute (hours)\n- The gold standard of test quality\n- \"Did the test VERIFY the code?\"", 5.2, 1.1, 4.3, 2.0, PURPLE);
addCard(s, "Recommendation", "Daily CI: code coverage (fast, 80% minimum)\nWeekly/nightly: mutation testing (thorough, find gaps)\nFocus mutation testing on critical modules first", 0.5, 3.3, 9, 1.2, GREEN);
addTagline(s, "Coverage says code was run. Mutation testing says it was verified.");

// 25: Chaos & Load Testing
s = contentSlide(pres, "Chaos & Load Testing");
addCard(s, "Chaos Testing", "Inject failures to verify resilience:\n- Kill random pods (Chaos Monkey)\n- Network latency/partition (Litmus)\n- CPU/memory pressure\n- DNS failures\n\nGoal: system degrades gracefully\nnot catastrophically", 0.5, 1.1, 4.3, 2.3, RED);
addCard(s, "Load Testing", "Verify performance under pressure:\n- Locust (Python, easy scripting)\n- k6 (JavaScript, developer-friendly)\n- JMeter (enterprise, GUI)\n\nTest scenarios:\n- Normal load baseline\n- Peak traffic simulation\n- Sustained load (soak test)\n- Spike test (sudden surge)", 5.2, 1.1, 4.3, 2.3, ORANGE);
addTagline(s, "If you haven't tested failure, you haven't tested");

// === SECTION 5: TDD WITH AI ===
sectionSlide(pres, "SECTION 05", "TDD with AI", "Red-Green-Refactor Accelerated");

// 27: TDD + AI Workflow
s = contentSlide(pres, "Test-Driven Development with AI");
addNumberedItem(s, 1, "Red: Write Test Spec", "Human defines WHAT to test (behavior, requirements)", 0.5, 1.1, 9, RED, RED);
addNumberedItem(s, 2, "Red: AI Generates Test", "AI writes the test code from your spec - review it!", 0.5, 1.75, 9, PURPLE, PURPLE);
addNumberedItem(s, 3, "Green: AI Implements", "AI writes minimal code to pass the test", 0.5, 2.4, 9, GREEN, GREEN);
addNumberedItem(s, 4, "Review Implementation", "Human verifies correctness, security, design", 0.5, 3.05, 9, BLUE, BLUE);
addNumberedItem(s, 5, "Refactor Together", "AI suggests refactoring, human approves changes", 0.5, 3.7, 9, ORANGE, ORANGE);

// 28: TDD + AI Example
s = contentSlide(pres, "TDD + AI: Practical Example");
addCard(s, "Step 1: Human Writes Spec", '"I need a UserService with:\n- create_user(name, email) -> User\n- Validates email format\n- Rejects duplicate emails\n- Hashes password before storing\n- Returns user without password field"', 0.5, 1.1, 4.3, 1.7, RED);
addCard(s, "Step 2: AI Generates Tests", "def test_create_user_success():\ndef test_create_user_invalid_email():\ndef test_create_user_duplicate_email():\ndef test_password_is_hashed():\ndef test_response_excludes_password():\ndef test_create_user_empty_name():", 5.2, 1.1, 4.3, 1.7, PURPLE);
addCard(s, "Step 3-5: AI Implements, Human Reviews, Both Refactor", "Human controls the design decisions, AI accelerates the implementation\nResult: well-tested code from the start, with human oversight at every step", 0.5, 3.0, 9, 1.4, GREEN);
addTagline(s, "Human controls WHAT, AI accelerates HOW");

// === SECTION 6: LAB ===
sectionSlide(pres, "SECTION 06", "Hands-on Lab", "Building a Test Suite with AI (90 min)");

// 30: Lab Overview
s = contentSlide(pres, "Lab: Comprehensive Testing with AI");
addCard(s, "Part 1: AI Unit Tests (30 min)", "1. Select a project module\n2. AI-generate comprehensive tests\n3. Review and correct each test\n4. Run: pytest --cov --cov-report=html\n5. Add manual tests for gaps", 0.5, 1.1, 2.8, 2.0, GREEN);
addCard(s, "Part 2: Integration + Property (30 min)", "1. API integration tests (pytest + requests)\n2. Hypothesis property-based tests\n3. AI suggest test properties\n4. Configure test database fixtures\n5. Verify with coverage report", 3.6, 1.1, 2.8, 2.0, BLUE);
addCard(s, "Part 3: CI + Mutation (30 min)", "1. Coverage enforcement in GH Actions\n2. pytest-xdist parallel execution\n3. mutmut mutation testing\n4. Kill surviving mutants\n5. Document testing strategy", 6.7, 1.1, 2.8, 2.0, PURPLE);

// 31: Lab Part 1 Detail
s = contentSlide(pres, "Lab Part 1: AI-Generated Unit Tests");
addCard(s, "Step 1: Select Module", "Choose a module from your capstone project\nIdeal: 5-10 functions, mix of logic\nExamples: user service, cart, validation\n\nCopy the source code for AI prompting", 0.5, 1.1, 4.3, 1.7, BLUE);
addCard(s, "Step 2: Generate & Review", "Prompt AI with source + requirements\nReview EVERY generated test:\n- Is the assertion meaningful?\n- Does it test behavior not implementation?\n- Are edge cases covered?\n- Would it catch a real bug?\n\nFix or discard bad tests", 5.2, 1.1, 4.3, 1.7, GREEN);
addCard(s, "Step 3: Run & Fill Gaps", "pytest --cov=myapp --cov-branch --cov-report=html\nOpen htmlcov/index.html - find uncovered branches\nWrite manual tests for: domain logic, security, concurrency", 0.5, 3.0, 9, 1.3, PURPLE);

// 32: Lab Part 2 Detail
s = contentSlide(pres, "Lab Part 2: Integration & Property Tests");
addCard(s, "API Integration Test", "@pytest.fixture\ndef client():\n    app.config['TESTING'] = True\n    with app.test_client() as c:\n        yield c\n\ndef test_create_and_get_user(client):\n    resp = client.post('/users',\n        json={'name': 'Alice', 'email': 'a@b.com'})\n    assert resp.status_code == 201\n    user_id = resp.json['id']\n    resp = client.get(f'/users/{user_id}')\n    assert resp.json['name'] == 'Alice'", 0.5, 1.1, 4.3, 2.8, BLUE);
addCard(s, "Property-Based Test", "from hypothesis import given, strategies as st\n\n@given(st.text(min_size=1, max_size=100))\ndef test_username_stored_correctly(name):\n    user = create_user(name=name, email='t@t.com')\n    assert user.name == name\n\n@given(st.emails())\ndef test_valid_emails_accepted(email):\n    user = create_user(name='test', email=email)\n    assert user.email == email\n\n# AI prompt: 'Suggest 5 properties for\n# my UserService that should always hold'", 5.2, 1.1, 4.3, 2.8, PURPLE);

// 33: Lab Part 3 Detail
s = contentSlide(pres, "Lab Part 3: CI Enforcement & Mutation");
addCard(s, "CI Configuration", "# .github/workflows/test.yml\njobs:\n  test:\n    steps:\n    - run: pip install -r requirements-test.txt\n    - run: pytest -n auto --cov=myapp \\\n        --cov-branch --cov-fail-under=80 \\\n        --cov-report=xml\n    - uses: codecov/codecov-action@v3", 0.5, 1.1, 4.3, 2.0, GREEN);
addCard(s, "Mutation Testing", "# Run mutation testing\nmutmut run --paths-to-mutate=myapp/ \\\n  --tests-dir=tests/\n\n# Check results\nmutmut results\n\n# For each surviving mutant:\nmutmut show <id>  # understand what changed\n# Write a test that catches it\n# Re-run to verify it's killed", 5.2, 1.1, 4.3, 2.0, RED);
addTagline(s, "Surviving mutants reveal where your tests are weakest");

// === SECTION 7: WRAP-UP ===
sectionSlide(pres, "SECTION 07", "Assessment & Next Steps", "Deliverables & Key Takeaways");

// 35: Assessment
s = contentSlide(pres, "Week 9 Assessment");
addCard(s, "Deliverables", "1. Unit + integration + property-based tests\n2. 80%+ branch coverage (enforced in CI)\n3. Mutation testing results & analysis\n4. CI pipeline with test enforcement\n5. AI collaboration log\n   (prompts used, tests accepted/rejected)", 0.5, 1.1, 4.3, 2.3, BLUE);
addCard(s, "Quality Criteria", "- Tests are meaningful (not tautological)\n- Edge cases covered (boundaries, null, empty)\n- Integration tests use real dependencies\n- Mutation score > 70% on critical modules\n- CI blocks merge on coverage drop\n- Clear documentation of testing strategy", 5.2, 1.1, 4.3, 2.3, PURPLE);

// 36: Recommended Reading
s = contentSlide(pres, "Recommended Reading");
addNumberedItem(s, 1, "Succeeding with Agile", "Cohn, M. (2009) - Test pyramid concept origin", 0.5, 1.1, 9, BLUE, BLUE);
addNumberedItem(s, 2, "Hypothesis Documentation", "hypothesis.readthedocs.io - Property-based testing in Python", 0.5, 1.75, 9, GREEN, GREEN);
addNumberedItem(s, 3, "Test Pyramid", "Fowler, M. (2012) - martinfowler.com - Classic reference", 0.5, 2.4, 9, PURPLE, PURPLE);
addNumberedItem(s, 4, "CodaMosa", "Lemieux et al. (2023) ICSE - AI-assisted test generation research", 0.5, 3.05, 9, ORANGE, ORANGE);
addTagline(s, "Testing is a skill - these resources will sharpen yours");

// 37: Key Takeaways
s = contentSlide(pres, "Key Takeaways");
addNumberedItem(s, 1, "Pyramid First", "Many unit tests, fewer integration, minimal E2E", 0.5, 1.1, 9, GREEN, GREEN);
addNumberedItem(s, 2, "AI Accelerates", "AI writes 80% of tests, human adds domain expertise", 0.5, 1.75, 9, BLUE, BLUE);
addNumberedItem(s, 3, "Review Everything", "AI tests need human validation - bad tests give false confidence", 0.5, 2.4, 9, RED, RED);
addNumberedItem(s, 4, "Enforce in CI", "80%+ coverage, parallel execution, block on regression", 0.5, 3.05, 9, PURPLE, PURPLE);
addNumberedItem(s, 5, "Mutation = Truth", "Mutation testing reveals real test quality, not just coverage", 0.5, 3.7, 9, ORANGE, ORANGE);

// 38: Next Week Preview
s = contentSlide(pres, "Next Week: MLOps & AI Model Deployment");
addCard(s, "Week 10 Preview", "- ML model lifecycle management\n- Model versioning & registry\n- Feature stores & pipelines\n- Model monitoring & drift detection\n- A/B testing for models\n- MLflow, Kubeflow, Seldon", 0.5, 1.1, 9, 2.0, PURPLE);
addTagline(s, "DevOps for machine learning - where software meets data science");

// 39: Q&A
s = pres.addSlide();
s.background = { color: BG };
s.addShape("oval", { x: 2.5, y: -0.5, w: 3, h: 3, fill: { color: "8B5CF6", transparency: 88 } });
s.addShape("oval", { x: 6, y: 3.5, w: 2.5, h: 2.5, fill: { color: "3B82F6", transparency: 88 } });
s.addText("Questions?", { x: 0, y: 1.8, w: 10, h: 0.7, fontSize: 36, fontFace: "Arial", bold: true, color: WHITE, align: "center" });
s.addText("Week 9: Automated Testing with AI", { x: 1, y: 2.7, w: 8, h: 0.4, fontSize: 14, fontFace: "Arial", color: BLUE, align: "center" });
s.addText("Anirach Mingkhwan | FITM, KMUTNB", { x: 2, y: 3.3, w: 6, h: 0.3, fontSize: 11, fontFace: "Arial", color: GRAY, align: "center" });
addTagline(s, '"Write tests that you\'d trust your career with"');

const outPath = "/home/clawdbot/clawd/tmp/Week09_raw.pptx";
pres.writeFile({ fileName: outPath }).then(() => {
  console.log(`Saved ${outPath} (${pres.slides.length} slides)`);
}).catch(err => console.error("Error:", err));
