// Copyright 2021 Associated Universities, Inc. Washington DC, USA.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#pragma once
#include "hpg_config.hpp"
#include "hpg_error.hpp"
#include "hpg_export.h"

#include <memory>
#include <tuple>
#if HPG_API >= 17
# include <variant>
#endif

namespace hpg {

/** type for containing a value or an error
 *
 * normally appears as a return value type for a method or function that can
 * fail
 */
#if HPG_API >= 17
template <typename T>
using rval_t = std::variant<Error, T>;
#else // HPG_API < 17
template <typename T>
using rval_t = std::tuple<std::unique_ptr<Error>, T>;
#endif // HPG_API >= 17

/** query whether rval_t value contains an error
 */
template <typename T>
HPG_EXPORT inline bool
is_error(const rval_t<T>& rv) {
#if HPG_API >= 17
  return std::holds_alternative<Error>(rv);
#else // HPG_API < 17
  return bool(std::get<0>(rv));
#endif // HPG_API >= 17
}

/** type trait to get type parameter of a template type
 */
template <typename T>
struct HPG_EXPORT rval_value {
  using type = void; /**< parameter type */
};
/** specialized type trait to get value type from rval_t type
 */
template <typename T>
struct rval_value<rval_t<T>> {
  using type = T; /**< parameter type */
};

/** query whether rval_t value contains a (non-error) value
 */
template <typename T>
HPG_EXPORT inline bool
is_value(const rval_t<T>& rv) {
#if HPG_API >= 17
  return std::holds_alternative<T>(rv);
#else // HPG_API < 17
  return !bool(std::get<0>(rv));
#endif // HPG_API >= 17
}

/** get value from an rval_t value
 */
#if __cplusplus >= 201402L
template <typename RV>
HPG_EXPORT inline auto
get_value(RV&& rv) {
  return std::get<1>(std::forward<RV>(rv));
}
#else // __cplusplus < 201402L
template <typename RV>
HPG_EXPORT inline const typename rval_value<RV>::type&
get_value(const RV& rv) {
  return std::get<1>(rv);
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&
get_value(RV& rv) {
  return std::get<1>(rv);
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&&
get_value(RV&& rv) {
  return std::get<1>(std::move(rv));
}
#endif // __cplusplus >= 201402L

/** get error from an rval_t value
 */
#if __cplusplus >= 201402L
template <typename RV>
HPG_EXPORT inline auto
get_error(RV&& rv) {
# if HPG_API >= 17
  return std::get<0>(std::forward<RV>(rv));
# else
  return *std::get<0>(std::forward<RV>(rv));
# endif
}
#else // __cplusplus < 201402L
template <typename RV>
HPG_EXPORT inline const typename rval_value<RV>::type&
get_error(const RV& rv) {
# if HPG_API >= 17
#  error "Unsupported c++ standard and HPG API version"
# else // HPG_API < 17
  return *std::get<0>(rv);
# endif // HPG_API >= 17
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&
get_error(RV& rv) {
# if HPG_API >= 17
#  error "Unsupported c++ standard and HPG API version"
# else // HPG_API < 17
  return *std::get<0>(rv);
# endif // HPG_API >= 17
}
template <typename RV>
HPG_EXPORT inline typename rval_value<RV>::type&&
get_error(RV&& rv) {
  return *std::get<0>(std::move(rv));
}
#endif // __cplusplus >= 201402L

/** create an rval_t value from a (non-error) value
 */
template <typename T>
HPG_EXPORT inline rval_t<T>
rval(T&& t) {
#if HPG_API >= 17
  return rval_t<T>(std::forward<T>(t));
#else // HPG_API < 17
  return {std::unique_ptr<Error>(), std::forward<T>(t)};
#endif // HPG_API >= 17
}

/** create an rval_t value from an Error
 */
template <typename T>
HPG_EXPORT inline rval_t<T>
rval(const Error& err) {
#if HPG_API >= 17
  return rval_t<T>(err);
#else // HPG_API < 17
  return {std::unique_ptr<Error>(new Error(err)), T()};
#endif // HPG_API >= 17
}

/** apply function that returns a plain value to an rval_t
 */
#if HPG_API >= 17 || defined(HPG_INTERNAL)
/** container class of static functions on rval_t values
 */
struct HPG_EXPORT RvalMM {

  /** apply function to value contained in an rval_t
   */
  template <typename T, typename F>
  static rval_t<std::invoke_result_t<F, T>>
  map(const rval_t<T>& rv, F f) {
    if (is_value(rv))
      return rval<std::invoke_result_t<F, T>>(f(get_value(rv)));
    else
      return rval<std::invoke_result_t<F, T>>(get_error(rv));
  }

  /** apply function to value contained in an rval_t
   */
  template <typename T, typename F>
  static rval_t<std::invoke_result_t<F, T>>
  map(rval_t<T>&& rv, F f) {
    if (is_value(rv))
      return rval<std::invoke_result_t<F, T>>(f(get_value(std::move(rv))));
    else
      return rval<std::invoke_result_t<F, T>>(get_error(std::move(rv)));
  }

  /** apply function that returns an rval_t value to an rval_t
   */
  template <typename T, typename F>
  static rval_t<typename rval_value<std::invoke_result_t<F, T>>::type>
  flatmap(const rval_t<T>& rv, F f) {
    if (is_value(rv))
      return f(get_value(rv));
    else
      return
        rval<typename rval_value<std::invoke_result_t<F, T>>::type>(
          get_error(rv));
  }

  /** apply function that returns an rval_t value to an rval_t
   */
  template <typename T, typename F>
  static rval_t<typename rval_value<std::invoke_result_t<F, T>>::type>
  flatmap(rval_t<T>&& rv, F f) {
    if (is_value(rv))
      return f(get_value(std::move(rv)));
    else
      return
        rval<typename rval_value<std::invoke_result_t<F, T>>::type>(
          get_error(std::move(rv)));
  }

  /** apply one of two functions, depending on contained value type, with common
   * result type
   */
  template <typename T, typename ValF, typename ErrF>
  static std::invoke_result_t<ValF, T>
  fold(const rval_t<T>& rv, ValF vf, ErrF ef) {
    static_assert(
      std::is_same_v<
        std::invoke_result_t<ValF, T>,
        std::invoke_result_t<ErrF, Error>>);

    if (is_value(rv))
      return vf(get_value(rv));
    else
      return ef(get_error(rv));
  }

  /** apply one of two functions, depending on contained value type, with common
   * result type
   */
  template <typename T, typename ValF, typename ErrF>
  static std::invoke_result_t<ValF, T>
  fold(rval_t<T>&& rv, ValF vf, ErrF ef) {
    static_assert(
      std::is_same_v<
        std::invoke_result_t<ValF, T>,
        std::invoke_result_t<ErrF, Error>>);

    if (is_value(rv))
      return vf(get_value(std::move(rv)));
    else
      return ef(get_error(std::move(rv)));
  }
};

/** base class for monad type class instances
 */
template <typename Derived, template <typename> typename M>
struct MonadBase {

  template <typename T>
  static M<T>
  flatten(const M<M<T>>& mmt) {
    return Derived::flat_map(mmt, [](auto&& mt) { return mt; });
  }

  template <typename T>
  static M<T>
  flatten(M<M<T>>&& mmt) {
    return Derived::flat_map(mmt, [](auto&& mt) { return std::move(mt); });
  }

  template <typename F>
  static auto
  lift(F f) {
    return
      [f]<typename MA>(MA&& ma) {
        return map(std::forward<MA>(ma), f);
      };
  }

  template <
    typename T,
    typename F,
    typename S = std::invoke_result_t<F, T>>
  static M<S>
  map(const M<T>& mt, F f) {
    return
      Derived::flat_map(
        mt,
        [f](auto&& t) {
          return Derived::pure(f(t));
        });
  }

  template <
    typename T,
    typename F,
    typename S = std::invoke_result_t<F, T>>
  static M<S>
  map(M<T>&& mt, F f) {
    return
      Derived::flat_map(
        std::move(mt),
        [f](auto&& t) {
          return Derived::pure(f(std::move(t)));
        });
  }

  // TODO: is this correct? There's a bias toward the first argument in
  // maintaining effects in the result.
  template <typename A, typename B>
  static M<std::tuple<A, B>>
  product(const M<A>& ma, const M<B>& mb) {
    return
      Derived::flat_map(
        ma,
        [mb](auto&& a) {
          return
            Derived::flat_map(
              mb,
              [a](auto&& b) {
                return std::make_tuple(a, b);
              });
        });
  }
};

/** monad type class
 */
template <template <typename> typename M>
struct Monad
  : public MonadBase<Monad<M>, M> {

  template <typename T>
  struct value {
    using type = void;
  };

  template <typename T>
  using value_t = typename value<T>::type;

  template <typename T>
  static M<T>
  pure(const T& t);

  template <
    typename T,
    typename F,
    typename S = value_t<std::invoke_result_t<F, T>>>
  static M<S>
  flat_map(const M<T>& t, F f);

  template <
    typename T,
    typename F,
    typename S = value_t<std::invoke_result_t<F, T>>>
  static M<S>
  flat_map(M<T>&& t, F f);

  template <
    typename A,
    typename F,
    typename B =
      std::variant_alternative_t<1, value_t<std::invoke_result_t<F, A>>>>
  static M<B>
  tail_rec_m(A&& a, F f);
};

/** monad type class instance for rval_t
 */
template <>
struct Monad<rval_t>
  : public MonadBase<Monad<rval_t>, rval_t> {

  template <typename T>
  using M = rval_t<T>;

  template <typename T>
  struct value {
    using type = void;
  };

  template <typename T>
  struct value<rval_t<T>> {
    using type = T;
  };

  template <typename T>
  using value_t = typename value<T>::type;

  template <typename T>
  static rval_t<T> pure(T&& t) {
    return rval<T>(std::forward<T>(t));
  }

  template <
    typename T,
    typename F,
    typename S = value_t<std::invoke_result_t<F, T>>>
  static rval_t<S>
  flat_map(const rval_t<T>& rv, F f) {
    return RvalMM::flatmap(rv, f);
  }

  template <
    typename T,
    typename F,
    typename S = value_t<std::invoke_result_t<F, T>>>
  static rval_t<S>
  flat_map(rval_t<T>&& rv, F f) {
    return RvalMM::flatmap(std::move(rv), f);
  }

  template <
    typename A,
    typename F,
    typename B =
      std::variant_alternative_t<1, value_t<std::invoke_result_t<F, A>>>>
  static rval_t<B>
  tail_rec_m(A&& a, F f) {
    auto mab = f(std::forward<A>(a));
    bool is_a = true;
    // be careful here to avoid anything but move construction/assignment of
    // values in mab
    while (is_value(mab) && is_a) {
      auto ab = get_value(std::move(mab));
      is_a = std::holds_alternative<A>(ab);
      if (is_a)
        mab = f(std::get<A>(std::move(ab)));
      else
        mab = std::move(ab);
    }
    if (is_value(mab))
      return rval<B>(std::get<B>(get_value(std::move(mab))));
    else
      return rval<B>(get_error(mab));
  }
};

/** type trait for types with a functor type class instance
 *
 * note that we're using the term "functor" in the sense of category theory or
 * functional programming, not the usual c++ terminology */
template <template <typename> typename F>
struct HPG_EXPORT functor {
  using type = void;
};

template <template <typename> typename F>
concept HasFunctor = requires {
  typename functor<F>;
};

/** type trait for types with an applicative type class instance */
template <template <typename> typename F>
struct HPG_EXPORT applicative
  : public functor<F> {
  using type = void;
};

template <template <typename> typename F>
concept HasApplicative = requires {
  typename applicative<F>;
};

/** type trait for types with a monad type class instance */
template <template <typename> typename M>
struct HPG_EXPORT monad
  : public applicative<M> {
  using type = void;
};

template <template <typename> typename M>
concept HasMonad = requires {
  typename monad<M>;
};

/** monad type class trait for rval_t */
template <>
struct HPG_EXPORT monad<rval_t> {
  using type = Monad<rval_t>;
};

/** functor type class trait for rval_t
 *
 * @todo find a way to avoid the repetition exhibited here for the monad and
 * functor type traits */
template <>
struct HPG_EXPORT functor<rval_t> {
  using type = Monad<rval_t>;
};

template <
  template <typename> typename M,
  typename A,
  typename B>
struct Kleisli;

template <
  template <typename, typename> typename KM,
  template <typename> typename M,
  typename A,
  typename B,
  typename F>
struct KleisliF;

/** data type for functions A => M\<B\>
 */
template <
  template <typename> typename M,
  typename A,
  typename B>
struct HPG_EXPORT Kleisli {

  template <typename A1, typename B1>
  using KM = Kleisli<M, A1, B1>;

  template <typename F>
  static KleisliF<KM, M, A, B, F>
  wrap(F&& f) {
    return KleisliF<KM, M, A, B, F>(std::forward<F>(f));
  }
};

/** data type for a given function A => M\<B\>
 */
template <
  template <typename, typename> typename KM,
  template <typename> typename M,
  typename A,
  typename B,
  typename F>
struct HPG_EXPORT KleisliF {

  /** wrapped function */
  F m_f;

  /** constructor */
  KleisliF(const F& f)
    : m_f(f) {}

  /** constructor */
  KleisliF(F&& f)
    : m_f(std::move(f)) {}

  /** evaluate the function for a value */
  M<B>
  run(const A& a) {
    return m_f(a);
  }

  /** evaluate the function for a value */
  M<B>
  run(A&& a) {
    return m_f(std::move(a));
  }

  /** compose this function with another Kleisli-like function
   *
   * @param g, a function B => M\<C\> for some C
   *
   * @return Kleisli instance for g(f)
   */
  template <typename G> requires HasMonad<M>
  auto
  and_then(G&& g) const & {
    using C =
      typename monad<M>::type::template value_t<std::invoke_result_t<G, B>>;

    return
      KM<A, C>::wrap(
        [g, f=m_f]<typename AA>(AA&& a) {
          return monad<M>::type::flat_map(f(std::forward<AA>(a)), g);
        });

  }

  /** compose this function with a plain function
   *
   * @param g, a function B => C for some C
   *
   * @return Kleisli instance for g(f) (understood to mean that g acts on the
   * value contained in the result of f)
   */
  template <typename G> requires HasFunctor<M>
  auto
  map(G&& g) const & {
    using C = std::invoke_result_t<G, B>;
    return
      KM<A, C>::wrap(
        [g, f=m_f]<typename AA>(AA&& a) {
          return functor<M>::type::map(f(std::forward<AA>(a)), g);
        });
  }
};

/** specialization of KleisliF for functions with no argument
 */
template <
  template <typename, typename> typename KM,
  template <typename> typename M,
  typename B,
  typename F>
struct HPG_EXPORT KleisliF<KM, M, void, B, F> {

  F m_f;

  KleisliF(const F& f)
    : m_f(f) {}

  KleisliF(F&& f)
    : m_f(std::move(f)) {}

  M<B>
  run() {
    return m_f();
  }

  template <typename G> requires HasMonad<M>
  auto
  and_then(G&& g) const & {
    using C =
      typename monad<M>::type::template value_t<std::invoke_result_t<G, B>>;

    return
      KM<void, C>::wrap(
        [g, f=m_f]() {
          return monad<M>::type::flat_map(f(), g);
        });
  }

  template <typename G> requires HasFunctor<M>
  auto
  map(G&& g) const & {
    using C = std::invoke_result_t<G, B>;
    return
      KM<void, C>::wrap(
        [g, f=m_f]() {
          return functor<M>::type::map(f(), g);
        });
  }
};

// The following classes may be of use in a functional programming style of
// error handling; otherwise, they can be safely ignored.

template <typename A, typename B, typename F>
struct HPG_EXPORT RvalMF;

/** functions with domain A and range rval_t<B> */
template <typename A, typename B>
struct HPG_EXPORT RvalM
  : public Kleisli<rval_t, A, B> {

  template <typename F>
  static RvalMF<A, B, F>
  wrap(F&& f) {
    return RvalMF<A, B, F>(std::forward<F>(f));
  }
};

/** a function with domain A and range rval_t<B> */
template <typename A, typename B, typename F>
struct HPG_EXPORT RvalMF
  : public KleisliF<RvalM, rval_t, A, B, F> {

  using KleisliF<RvalM, rval_t, A, B, F>::m_f;

  /** constructor */
  RvalMF(const F& f)
    : KleisliF<RvalM, rval_t, A, B, F>(f) {}

  /** constructor */
  RvalMF(F&& f)
    : KleisliF<RvalM, rval_t, A, B, F>(std::move(f)) {}

  template <typename G>
  auto
  and_then_loop(unsigned n, G&& g) const & {
    return
      KleisliF<RvalM, rval_t, A, B, F>::and_then(
        [g, n]<typename BB>(BB&& b) {
          using ibb = std::variant<std::tuple<unsigned, B>, B>;
          return
            Monad<rval_t>::tail_rec_m(
              std::make_tuple(n, std::forward<BB>(b)),
              [g, n](auto&& i_b) -> rval_t<ibb> {
                auto& [i, b] = i_b;
                if (i == 0)
                  return ibb(std::move(b));
                else
                  return
                    Monad<rval_t>::map(
                      g(n - i, std::move(b)),
                      [i1=i - 1](auto&& b) -> ibb {
                        return std::make_tuple(i1, std::move(b));
                      });
              });
        });
  }

  template <typename G>
  auto
  and_then_repeat(unsigned n, G&& g) const & {
    return
      and_then_loop(n, [g](unsigned, auto&& b) { return g(std::move(b)); });
  }
};

/** a function with empty domain and range rval_t<B> */
template <typename B, typename F>
struct HPG_EXPORT RvalMF<void, B, F>
  : public KleisliF<RvalM, rval_t, void, B, F> {

  using KleisliF<RvalM, rval_t, void, B, F>::m_f;

  /** constructor */
  RvalMF(const F& f)
    : KleisliF<RvalM, rval_t, void, B, F>(f) {}

  /** constructor */
  RvalMF(F&& f)
    : KleisliF<RvalM, rval_t, void, B, F>(std::move(f)) {}

  template <typename G>
  auto
  and_then_loop(unsigned n, G&& g) const & {
    return
      KleisliF<RvalM, rval_t, void, B, F>::and_then(
        [g, n]<typename BB>(BB&& b) {
          using ibb = std::variant<std::tuple<unsigned, B>, B>;
          return
            Monad<rval_t>::tail_rec_m(
              std::make_tuple(n, std::forward<BB>(b)),
              [g, n](auto&& i_b) -> rval_t<ibb> {
                auto& [i, b] = i_b;
                if (i == 0)
                  return ibb(std::move(b));
                else
                  return
                    Monad<rval_t>::map(
                      g(n - i, std::move(b)),
                      [i1=i - 1](auto&& b) -> ibb {
                        return std::make_tuple(i1, std::move(b));
                      });
              });
        });
  }

  template <typename G>
  auto
  and_then_repeat(unsigned n, G&& g) const & {
    return
      and_then_loop(n, [g](unsigned, auto&& b) { return g(std::move(b)); });
  }
};

#endif // HPG_API >= 17

}  // end namespace hpg

// Local Variables:
// mode: c++
// c-basic-offset: 2
// fill-column: 80
// indent-tabs-mode: nil
// End:
