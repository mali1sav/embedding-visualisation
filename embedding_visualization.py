def batch_scrape_urls(urls: List[str], max_urls: int = 10) -> List[Dict]:
    """
    Batch scrape multiple URLs using Firecrawl's scraping endpoint.
    Limited to max_urls during testing phase.
    """
    try:
        # Limit number of URLs during testing
        urls = urls[:max_urls]
        st.write(f"Batch scraping {len(urls)} URLs...")
        
        # Initialize parameters for scraping according to API docs
        params = {
            'formats': ['markdown', 'html'],
            'onlyMainContent': True,
            'actions': [
                {"type": "wait", "milliseconds": 2000},  # Initial wait for page load
                {"type": "scroll", "pixels": 500},       # Scroll to load dynamic content
                {"type": "wait", "milliseconds": 1000}   # Wait after scroll
            ]
        }
        
        # Process URLs sequentially with rate limiting
        results = []
        for url in urls:
            try:
                st.write(f"Processing URL: {url}")
                
                # Make the request using scrape_url
                result = firecrawl.scrape_url(url, params)
                
                if result and isinstance(result, dict):
                    content = result.get('data', {}).get('markdown', '')
                    metadata = result.get('metadata', {})
                    
                    if content:
                        st.success(f"Successfully scraped: {url}")
                        results.append({
                            'url': url,
                            'markdown': content,
                            'title': metadata.get('title', ''),
                            'meta_description': metadata.get('description', ''),
                            'publish_date': metadata.get('published_time')
                        })
                    else:
                        st.warning(f"No content found for {url}, using mock data")
                        results.append(generate_mock_crypto_content(url))
                else:
                    st.warning(f"Invalid response for {url}, using mock data")
                    results.append(generate_mock_crypto_content(url))
                
            except Exception as e:
                st.error(f"Error scraping {url}: {str(e)}")
                results.append(generate_mock_crypto_content(url))
            
            time.sleep(2)  # Rate limiting between requests
        
        return process_batch_results(results, urls)
        
    except Exception as e:
        st.error(f"Error in batch scraping: {str(e)}")
        return [generate_mock_crypto_content(url) for url in urls]
