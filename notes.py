data = [0.95, 2.34, 1.78, 3.67, 2.45, 4.12, 3.89, 5.67, 4.56, 6.78,
        5.23, 7.89, 6.34, 8.45, 7.12, 9.67, 8.91, 10.23, 9.34, 11.45]
idx_range = (0, 19)
target_size = 20


"""

Currently we have a system that works as so:
    -   a subarray is created which represents the data between the provided indcies
    -   now an array is created full of the indexes of the subarray
    -   an additional array is created, using linspace, that creates an array of smashed or expanded indcies
    
    now for every target index, a loop iteration is run
        -   a lower and upper bound is generated 
        -   by looking at the target index and its neighbors, 
        -   then taking those values and finding integers that represent the range
            -   these numbers are the indexes in the original function that could be mapped to the index
        -   the max value of these indexes are taken, and assigned to the index
    
            
        Subarray Extracted: Entire data list [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
            Mapping:
            Target index 0 → uses indices [0, 1, 2] → max is 4
            Target index 1 → uses indices [0, 1, 2, 3, 4] → max is 5
            Target index 2 → uses indices [3, 4, 5, 6] → max is 9
            Target index 3 → uses indices [5, 6, 7, 8, 9] → max is 9
            Target index 4 → uses indices [7, 8, 9] → max is 6
        Output: [4, 5, 9, 9, 6]

    
    
    What I would do
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
        to a size of 5, I would take every grouping of 2, bc 10/5 = 2, then I would take the average of it, and make that the value
        3+1 = 4/2 = 2
        4+1/2
        5+9/2
        
        size 11 ->5
        
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 7]
         0  1  2  3  4  5  6  7  8  9  10
        
        I could divide 11/5 = 2.2, then take all of the energy in every 2 block, and an aditional .2 of energy from surounding blocks? 
        
        so first would be  0 - 2.2 
            so 3,1 and .2 of 4
            so index 0,1, and .2 of 2
            3 + 1 + (.2*4) / 2.2
            with weights 1 + 1 + .2 = 2.2
        then 2.2 - 4.4
            so .8 of 4, all of 1 and .4 of 5 
            so .8 of index 3, index 4 and .4 of index 5
        4.4 - 6.6
        
        
        
            
            
        Considering repeating or irrational numbers, could be tricky but maybe not
        say we want to map 10 to 3
        10/3 = 3.3...
        so we take first 3 then .3 of 4th, then we take .6 of 4th and so on
        the concern comes from when we send the loop, but the loop only runs target indices times so its fine,
        there is also the fact that if we increament using the ratio, errors will acumulate on larger data, because if we are using addition
            but first run can be 0* ratio -> 1* ratio, 1x to 2x... (n-1)x -> nx
            
            
            
        
        pyso do:
        
        Take original size, divide it by target size, the result is your ratio
        
        for every space in the resized data, 
        calculate the lower and upper bound representing the data you want
            0* ratio -> 1* ratio, 1x to 2x... (n-1)x -> nx
        
        these bounds represent the energy in the original signals indexes we want to extract 
        
         on the lower bound, we take the fractional component, potentially using fractional, integer = math.modf(x)
        this represents how much to take of that index 
        2.2 -4.4 takes .8 of the first
        
        now we need to determine how many indexes in the middle
         
        
        ok slow down there
        
        how does this handle a vereity of cases?
        
        1 1 100 2 4 5 
        6->3
        6/3=2
        1 51 4.5
        
        What if we use the square? 
        
        1 1 10000 4 16 25
        1 5002 28
        1 70 5.2
        
        ... maybe
        
        0 0 50 0 0 0
        6/3 = 2
        
        0 25 0 
        
        
        Using MAX
        -------------------------------------------------------------
        1 1 100 2 4 5 
        1 100 5 
        
        I mean this is my favorite result, but then how do we deal with decimals
        
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 7]
         0  1  2  3  4  5  6  7  8  9  10
         
         11->5 ratio 2.2
         
         0 -> 2.2
         0 1 2 3*.2 or 0 1 2*.2... I think its 3... no its 2, the ratio is the number of indices to check per iteration
         
         3 1 4
         3 1 .2*4 -> 3
         
         2.2-4.4
         2*.8 3 4
         4 1 5 
         .8*4 1 .4*5 = 3.2
         
         This seems fine but lets try with extreme cases 
         
         1 0 100 0 1
         0 1  2  3 4
         
         
         5->2 = 2.5
         
         0-2.5
         1 0 100*.5 = 50
         
         2.5-5
         100*.5 0 1 = 50
         
         50 50
         
         So, I feel like this is a perfect example of the limitations of the current system, we have a decision
         Either split this bin, or set it to one side, both of which suck
         In the real world with larger signals though I wonder how much of a problem this is, feel like it will be rare to have an exact split, and much more likely to have a partial
         
         0 0 100 0 0 0 0 0 0 0 0
         0 1 2 3 4 5 6 7 8 9 10
         
         11/5
         2.2
         
         0 0 .2* 100 = 20
         .8*100 0 0 = 80
         
         20 80 0 0 0
         
         This is probably a more real world senario, but we still have the issue
         
         I wonder if there is any information to gain from the phase of the signal as well, 
         With re reassignment method we can target what point of the bin out energy is located
         What if within the bins, the signal is a mystery to us manipulating it, but when it is reconstructed, it is complete, meaning if we manipulate it consistantly , it is fine
         
         How does stft handle frequencies that are on the edge of bins?? 
         
         LIMITATIONS OF CURRENT ALGO:
         
                ok.. So, when we take the original stft, the windowing function smears the data, according to the windows frequency response, I believe this is how the plain stft is able to deduce frequencies in the middle of bins,
                this is something you many need to consider and could be a way to upgrade the warping
                
                additionaly, consider how you are warping phase, I fear you may be mangling it, an option to manually set phase to zero could be an interesting distortion
                
        
        
        REAL PYSODO
        
        
        we need the original array
        
        divide the size of OG by new size to get ratio
        
        for every space in the new array, of size new size, do this:
                
                calculate relevant indices with current*ratio and (current+1)*ratio
                these indices may be floats
                the ratio is the number of indices to check per iteration
                traverse these indices between lower index and upper index
                
                        this is tricky because decimals matter
                                get your start index by taking the whole number of lower bound
                                increment until you reach the whole number of upper bound
                                        then use the decimal of the upper bound for final ratio
                                        
                                ok this is bs
                                        we want to increament ratio times
                                        calc bounds
                                        separate whole and decimal components
                                        
                                        i = int(lowerBound)
                                        while i <= int(upperBound) 
                                                if i == lowerbound
                                                        do first stuff
                                                elif i == number of Int(ratio(last))
                                                        do last stuff
                                                else 
                                                        do middle stuff
                                        max(current max, current index with ratio) 
                                        
                                but then how do we get the index, i + int(lower bound)? test case where decimal increments whole number? can that even happen? i dont think so, but like ratio = 2.9 
                                ratio = 2.9 
                                19/10
                                
                                 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
                                 
                                while i < 2 
                        1        0  - 1.9 
                                start = 0 end 1
                                        0 1
                                        0 1.9
                                1.9 - 3.8 
                                start = 1 end 3
                                        
                                3.8 - 5.7 
                                start 3 end 5
                                         
                                ....
                                5.7
                                7.6
                                9.5
                                11.4
                                13.3
                                15.2
                                17.1
                                19
                                
                                17.1 -19
                                start 17 end 19 
                                17*.9 18 !!19!! this is access 1.9 of the 19 data, which is correct, 19 will have no decimal, so then we dont even want to access the 19th var, bc it would be *0
                                when dealing with ratio * targetSize, so 1.9 * 10 it could be bad if 1.9 is infinitely repeating, 
                                        We can fix this by recalculating the ratio
                                        But, this may not be an issue 
                                        
                                
                        
                        
                        for first index, 1-the decimal is the multiplication factor for the value at its whole numbers index
                                0.0 - 2.2
                                1 - .0 = 1
                                1*index0 is first value to store
                                
                                2.2 - 4.4
                                1-.2 = .8 multiplication factor
                        then, we traverse the rest of the whole numbers, these have no multiplication factor
                                if(currentIndex+1 <(=?) upperIndex 
                                        we have a whole number
                        
                        then when we reach the last index, 
                                we take the decimal of the last upper index
                                
                                
                11/5 = 2.2
                
                0 1 2 3 4 5 6 7 8 9 10
                
                        indexes
                0-2.2
                        0
                        1
                        2*.2   
                2.2 -4.4
                        2*.8
                        3
                        4*.4        
                4.4-6.6
                        4*.6
                        5
                        6..
                6.6-8.8
                        6..
                        7
                        8..
                8.8-11
                        8..
                        9
                        10..
                        
                        
                2 -> 4
                
                [1,2]
                
                2/4 = .5 
                
        0 -.5
        
        start = 0, end = 0 
        .5 * value 1 = .5 which is max
        
        
        .5 - 1 
        0s  1e
        .5* 1 or 2 =2
        
        1 - 1.5
        1s 1e
        2 or .5 *2 = 2
        
        1.5 - 2
        
        1
        
        EXPANTIONNNNNNNNN
        
        
        so if i have an array of 2,2
        to 5 it would be
        0 2 0 2 0
        to 4 it would be 0 2 0 2
        1 1 1 to 10 would be
        
        0 1 0 0 0 1 0 0 1 0 
        0 1 2 3 4 5 6 7 8 9
        
        well like before, we can calculate a ratio
        
        10/3 = 3.33, so ideally, we would place an element every 3.33 indcies 
        
        it seems the thing i really care about is the spacing between concrete values, I want it to be centered well I guess too
        
        lets start with very straight forward cases
        
        [1,1]
        -> size 5
        [0,1,0,1,0]
        
        is very straight forward
        
        now how would we want
        [1,1]
        -> size 6: 
        [0 1 0 0 1 0]
        to 6 to behave
        
        [0 1 0 0 1 0]?
        
        or 0 1 0 1 0 0
        
        I would say the first case certainly, cemetric, fills nicely
        
        now 1 1 to 7
        
        [1,1]
        -> to size 7
        
        0 1 0 0 0 1 0 
        or
        0 0 1 0 1 0 0 
        or
        0 1 0 0 1 0 0 
        
        Like I said earlier I think I really just need equal spacing between frequencies, so i think the last result would be the most accurate, I want it to linearly streach
        
        1 1 1 -> 10
        
        0 0 0 3 0 0 3 0 0 3 is best?
        0 1 2 3 4 5 6 7 8 9
        its not centered
        you could minus certain value but this wont help with tightly packed examples like below
        
        10/3 = 3.333... 
        3.333
        6.666
        9.999
        
        
        every third index, insert
        
        10/6 = 1.6, every 2nd insert
        
         0 1 0 1 0 1 0 1 0 1
         
         ahh, tricky 
         
         10/6 = 1.6, every 2nd insert
         1 0 1 0 1 0 1 0 1 1 !!!!!!!!!!!!!!!!!!!!!!!!!!
         0 1 2 3 4 5 6 7 8 9 
         
         would feel the best
         
         1.6
         3.2
         4.8
         6.4
         8 
         9.6
         
         0 1 0 1 0 1 1 0 1 1 
         0 1 2 3 4 5 6 7 8 9 
         
         start with value at index 0, then use other method to distribute
         
         9/5 = 
         Step 1: 1.8
        Step 2: 3.6
        Step 3: 5.4
        Step 4: 7.2
        Step 5: 9.0
         1 0 1 0 1 1 0 1 0 1
         0 1 2 3 4 5 6 7 8 9 
         
         I mean it sure looks like a great distribution but it would nessesarily look like this for all values?
         
         what i have been trying to optimze for least amount of data touching each other, get the most spread between values possible
         how could spread be even calculated?
         
         
         So I think theres one way of thinking of this, and that is distribution
         we also can break this into more parts
                We have the whole distributation, 
                and the remainder distributation
        
        we can justify the information left right and center
        
        10/3 has a remainder of 1
        
        15/ 6 has a remainer is 2 remainder 3
        
        middle justified
         0 0 1 0 1 0 1 0 1 0 1  0  1  0  0 
         0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 
        
        15 <- 7 has 8 zeros
        
        
        Back to this example
        
        10/6 = 1.6, every 2nd insert
         1 0 1 0 1 0 1 0 1 1 !!!!!!!!!!!!!!!!!!!!!!!!!!
         0 1 2 3 4 5 6 7 8 9 
        
        also back to the idea of least touching
        
        we have 6 values, and 4 zeros
        4 zeros can separate 5 values
        
        with one remainder
        
        now I can decided where to put this value
                back to distribution, do I want it on the side or in the middle, well thats silly, we arnt only dealing with ones here
                well actually not that silly, we still need to decide
                
        1 2 3 4 5 6
        
        Well maybe I'm just being silly, seems like all of there distortions are equally bad 
        
        target size = 7
        size = 2
        
        7/2 = 3.5
        
        0 0 0 1 0 0 1
        1 2 3 4 5 6 7 
        
        4 -> 9
        9/4 = 
        
        2.25
        4.5
        6.75
        9
        
        0 1 0 0 1 0 1 0 1 
        1 2 3 4 5 6 7 8 9
        
        PYSODO
        
        ratio = targetSize / size
        
        for i in size
                newIndex = i*ratio
                newArray[round(newIndex)-1] = ogArray[i]
        
        
        
        
        NEXT STEPS
                look into this padding thing
                really getting to the point where we need organization
                make slight distortions SLIGHT, use talking audio as its very easy to distort, we want it to make small changes for small things
        
        
        
        
        
        
        
        
        
        
        
                
                
         
         
        Zero-Padding for Increased Resolution:

Method: Insert zeros into the time-domain signal before performing the STFT to increase the frequency resolution.
Consideration: While this technique enhances frequency resolution, it doesn't add new information but interpolates existing data.
Reference: Increasing STFT Resolution by Repeating the Window? | DSP Stack Exchange
        
    
    
    


"""