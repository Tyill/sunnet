# How to contribute

No external dependencies: boost, opencv, etc.

## Modules allowed for commits
 
 * [sunnet](https://github.com/Tyill/sunnet/tree/master/src/sunnet) - only fix bug. Interface no changed
 * [snOperator](https://github.com/Tyill/sunnet/tree/master/src/snOperator) - only new operator and fix bug
 * [python interface](https://github.com/Tyill/sunnet/tree/master/python/libsunnet) - no limits
 * [cpp interface](https://github.com/Tyill/sunnet/tree/master/cpp) - without restrictions, but with the agreement
 * [c_sharp interface](https://github.com/Tyill/sunnet/tree/master/c_sharp/libsunnet) - no limits
 * [example](https://github.com/Tyill/sunnet/tree/master/example) - no limits
 * [wiki](https://github.com/Tyill/sunnet/wiki) - no limits, but reliably
 
## Modules forbidden for commits

 * [snEngine](https://github.com/Tyill/sunnet/tree/master/src/snEngine)
 * [snBase](https://github.com/Tyill/sunnet/tree/master/src/snBase)
 
## Coding conventions

 * no any Template<> (only for native cpp)
 * camelCase for all names (except constants).
 

## Submitting changes

Please send a [GitHub Pull Request to sunnet](https://github.com/Tyill/sunnet/pull/new/master) 
with a clear list of what you've done (read more about [pull requests](https://help.github.com/articles/proposing-changes-to-your-work-with-pull-requests/). 


Always write a clear log message for your commits. One-line messages are fine for small changes, but bigger changes should look like this:

    $ git commit -m "A brief summary of the commit
    > 
    > A paragraph describing what changed and its impact."
	
	