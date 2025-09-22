import asyncio
import aiohttp
import time
import statistics
from datetime import datetime
import json

class APIStressTest:
    def __init__(self, base_url, project_id, num_users=20):
        self.base_url = base_url.rstrip('/')
        self.project_id = project_id
        self.num_users = num_users
        self.endpoint = f"{self.base_url}/api/v1/nlp/index/answer/{self.project_id}"
        self.results = []
        
    async def make_request(self, session, user_id, request_data):
        """Make a single API request and record metrics"""
        start_time = time.time()
        try:
            async with session.post(
                self.endpoint,
                json=request_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                # Try to read response body
                try:
                    response_body = await response.text()
                except:
                    response_body = "Unable to read response"
                
                result = {
                    'user_id': user_id,
                    'status_code': response.status,
                    'response_time': response_time,
                    'timestamp': datetime.now().isoformat(),
                    'success': 200 <= response.status < 300,
                    'response_size': len(response_body) if response_body else 0,
                    'error': None
                }
                
                return result
                
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            result = {
                'user_id': user_id,
                'status_code': 0,
                'response_time': response_time,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'response_size': 0,
                'error': str(e)
            }
            
            return result
    
    async def user_session(self, session, user_id, requests_per_user=5):
        """Simulate a user making multiple requests"""
        user_results = []
        
        # Sample request data - you can customize this
        sample_requests = [
            {"text": f"Hello from user {user_id}", "limit": 5},
            {"text": f"Test query {user_id} - request 2", "limit": 10},
            {"text": f"Search functionality test user {user_id}", "limit": 3},
            {"text": f"Performance test query {user_id}", "limit": 7},
            {"text": f"Final test from user {user_id}", "limit": 5}
        ]
        
        for i in range(requests_per_user):
            request_data = sample_requests[i % len(sample_requests)]
            result = await self.make_request(session, user_id, request_data)
            user_results.append(result)
            
            # Small delay between requests (optional)
            await asyncio.sleep(0.1)
        
        return user_results
    
    async def run_stress_test(self, requests_per_user=5, timeout=30):
        """Run the stress test with concurrent users"""
        print(f"Starting stress test:")
        print(f"- Endpoint: {self.endpoint}")
        print(f"- Concurrent users: {self.num_users}")
        print(f"- Requests per user: {requests_per_user}")
        print(f"- Total requests: {self.num_users * requests_per_user}")
        print("-" * 50)
        
        start_time = time.time()
        
        # Create session with timeout
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            # Create tasks for all users
            tasks = [
                self.user_session(session, user_id, requests_per_user) 
                for user_id in range(1, self.num_users + 1)
            ]
            
            # Run all user sessions concurrently
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Flatten results
        for user_results in all_results:
            if isinstance(user_results, list):
                self.results.extend(user_results)
            else:
                # Handle exceptions
                print(f"Error in user session: {user_results}")
        
        self.analyze_results(total_duration)
    
    def analyze_results(self, total_duration):
        """Analyze and display test results"""
        if not self.results:
            print("No results to analyze!")
            return
        
        successful_requests = [r for r in self.results if r['success']]
        failed_requests = [r for r in self.results if not r['success']]
        response_times = [r['response_time'] for r in successful_requests]
        
        print("\n" + "="*60)
        print("STRESS TEST RESULTS")
        print("="*60)
        
        print(f"Test Duration: {total_duration:.2f} seconds")
        print(f"Total Requests: {len(self.results)}")
        print(f"Successful Requests: {len(successful_requests)}")
        print(f"Failed Requests: {len(failed_requests)}")
        print(f"Success Rate: {(len(successful_requests)/len(self.results)*100):.2f}%")
        
        if response_times:
            print(f"\nResponse Time Statistics:")
            print(f"- Average: {statistics.mean(response_times):.3f}s")
            print(f"- Median: {statistics.median(response_times):.3f}s")
            print(f"- Min: {min(response_times):.3f}s")
            print(f"- Max: {max(response_times):.3f}s")
            
            if len(response_times) > 1:
                print(f"- Std Dev: {statistics.stdev(response_times):.3f}s")
        
        print(f"\nThroughput: {len(successful_requests)/total_duration:.2f} requests/second")
        
        # Status code breakdown
        status_codes = {}
        for result in self.results:
            status = result['status_code']
            status_codes[status] = status_codes.get(status, 0) + 1
        
        print(f"\nStatus Code Breakdown:")
        for status, count in sorted(status_codes.items()):
            print(f"- {status}: {count} requests")
        
        # Show errors if any
        if failed_requests:
            print(f"\nError Details:")
            error_summary = {}
            for result in failed_requests:
                error = result['error'] or f"HTTP {result['status_code']}"
                error_summary[error] = error_summary.get(error, 0) + 1
            
            for error, count in error_summary.items():
                print(f"- {error}: {count} occurrences")
    
    def save_detailed_results(self, filename="stress_test_results.json"):
        """Save detailed results to JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                'test_config': {
                    'endpoint': self.endpoint,
                    'num_users': self.num_users,
                    'timestamp': datetime.now().isoformat()
                },
                'results': self.results
            }, f, indent=2)
        print(f"\nDetailed results saved to: {filename}")

async def main():
    # Configuration - UPDATE THESE VALUES
    BASE_URL = "http://localhost:5000"  # Replace with your actual API base URL
    PROJECT_ID = "wer"  # From your screenshot
    NUM_CONCURRENT_USERS = 1
    REQUESTS_PER_USER = 1  # Each user will make 5 requests
    
    # Create and run stress test
    stress_test = APIStressTest(BASE_URL, PROJECT_ID, NUM_CONCURRENT_USERS)
    await stress_test.run_stress_test(REQUESTS_PER_USER)
    
    # Save detailed results
    stress_test.save_detailed_results()

if __name__ == "__main__":
    # Run the stress test
    print("API Stress Test Tool")
    print("Make sure to update BASE_URL and PROJECT_ID in the script!")
    print()
    
    asyncio.run(main())